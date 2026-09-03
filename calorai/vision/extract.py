"""Photo -> structured food items.

Runs on a separate vision model (Gemma 4 by default). Its only job is to
*extract* - it never decides or logs. The output, with a confidence per item,
is handed to the text agent, which applies memory and any caption and decides
whether to log or ask.

Gemma 4 on Sarvam accepts images only as base64 data URIs, so the file is read,
downsized, and inlined here.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from calorai.models.gateway import as_structured, get_vision_model

_MAX_EDGE = 1024
_MAX_BYTES = 20_000_000       # reject files larger than this before decoding
_MAX_PIXELS = 40_000_000      # ~decompression-bomb guard (Pillow default is 178M)
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


class VisionItem(BaseModel):
    name: str = Field(description="the food, as specific as you can be")
    quantity: float = Field(default=1.0, gt=0, le=100, description="estimated number of units on the plate")
    unit: str = Field(default="serving", description="piece, cup, bowl, slice, glass, plate...")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="0-1: how sure you are")

    @field_validator("quantity")
    @classmethod
    def _finite_q(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("quantity must be a positive finite number")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError("name must not be blank")
        return v.strip()


class _VisionOutput(BaseModel):
    is_food: bool = Field(description="false if the photo is not of food/drink")
    items: list[VisionItem] = Field(default_factory=list)
    note: str = Field(default="", description="anything that made the photo hard to read (blur, lighting, angle)")


@dataclass
class VisionResult:
    is_food: bool
    items: list[dict] = field(default_factory=list)
    note: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def low_confidence_items(self) -> list[str]:
        return [i["name"] for i in self.items if i.get("confidence", 1.0) < 0.5]

    def to_context(self) -> dict:
        """Compact form for the agent's prompt."""
        return {
            "is_food": self.is_food,
            "items": self.items,
            "note": self.note,
            "error": self.error,
        }


_PROMPT = """\
You are looking at a photo a user sent to log what they ate.

List every distinct food or drink item you can see. For each: a specific name, an
estimated portion (quantity + unit), and a confidence from 0 to 1.

- If you can't identify an item, still include it with a descriptive name
  ("unknown fried item") and a low confidence.
- If the photo is not food or drink at all, set is_food to false and leave items empty.
- Put anything that makes the photo hard to read (blur, lighting, odd angle) in "note".
Do not guess wildly - low confidence is better than a confident wrong answer.
"""


def _load_data_uri(path: Path) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        w, h = opened.size
        if w * h > _MAX_PIXELS:
            raise ValueError("image has too many pixels")
        opened = ImageOps.exif_transpose(opened)  # honour camera rotation
        if opened.mode in ("P", "RGBA", "LA"):
            # flatten transparency onto white so it doesn't become black in JPEG
            rgba = opened.convert("RGBA")
            image = Image.new("RGB", rgba.size, (255, 255, 255))
            image.paste(rgba, mask=rgba.split()[-1])
        else:
            image = opened.convert("RGB")

    image.thumbnail((_MAX_EDGE, _MAX_EDGE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _run_vision(content: list[dict]) -> _VisionOutput:
    """The one live call. Seam for tests to stub."""
    from langchain_core.messages import HumanMessage

    return as_structured(get_vision_model(), _VisionOutput).invoke([HumanMessage(content=content)])


def extract_food_from_image(image_path: str, caption: str | None = None) -> VisionResult:
    """Extract food items from a photo. Never raises - problems come back in
    `VisionResult.error`."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        return VisionResult(is_food=False, error="no image file at that path")
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        return VisionResult(is_food=False, error=f"{path.suffix or 'that'} is not a supported image type")

    size = path.stat().st_size
    if size > _MAX_BYTES:
        return VisionResult(is_food=False, error=f"image is too large ({size // 1_000_000} MB, limit {_MAX_BYTES // 1_000_000} MB)")

    try:
        data_uri = _load_data_uri(path)
    except Exception:
        return VisionResult(is_food=False, error="could not read that image")

    prompt = _PROMPT
    if caption:
        prompt += f'\nThe user also wrote: "{caption}". Use it to identify items, not to change portions.'

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]

    try:
        result = _run_vision(content)
    except Exception:
        return VisionResult(is_food=False, error="the vision model could not be reached")

    return VisionResult(
        is_food=result.is_food,
        items=[i.model_dump() for i in result.items],
        note=result.note,
    )
