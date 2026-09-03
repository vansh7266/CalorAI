"""Vision extraction + image-path routing tests. The vision model is stubbed."""

from __future__ import annotations

import pytest
from PIL import Image

from calorai.agent.graph import _after_ingest, _vision_extract
from calorai.agent.prompts import build_system_prompt
from calorai.vision import extract as vx
from calorai.vision.extract import VisionItem, _VisionOutput, extract_food_from_image


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "plate.jpg"
    Image.new("RGB", (200, 200), (200, 180, 120)).save(path)
    return str(path)


def _stub(model_output: _VisionOutput):
    class _S:
        def invoke(self, _messages):
            return model_output

    return lambda *a, **k: _S()


def test_missing_file_and_bad_type():
    assert extract_food_from_image("/no/such/file.jpg").error is not None
    assert "not a supported image type" in extract_food_from_image(__file__).error


def test_extract_parses_items(monkeypatch, photo):
    monkeypatch.setattr(
        vx,
        "as_structured",
        _stub(
            _VisionOutput(
                is_food=True,
                items=[
                    VisionItem(name="dal", quantity=1, unit="bowl", confidence=0.9),
                    VisionItem(name="unknown fried item", quantity=2, unit="piece", confidence=0.3),
                ],
                note="a bit dim",
            )
        ),
    )
    result = extract_food_from_image(photo)
    assert result.ok and result.is_food
    assert [i["name"] for i in result.items] == ["dal", "unknown fried item"]
    assert result.low_confidence_items == ["unknown fried item"]


def test_extract_not_food(monkeypatch, photo):
    monkeypatch.setattr(vx, "as_structured", _stub(_VisionOutput(is_food=False, note="looks like a car")))
    result = extract_food_from_image(photo)
    assert result.ok and not result.is_food


def test_extract_model_failure(monkeypatch, photo):
    def _boom(*a, **k):
        class _S:
            def invoke(self, _m):
                raise RuntimeError("api down")

        return _S()

    monkeypatch.setattr(vx, "as_structured", _boom)
    result = extract_food_from_image(photo)
    assert not result.ok and "vision model failed" in result.error


def test_prompt_renders_vision_block():
    text = build_system_prompt(
        vision_result={
            "is_food": True,
            "items": [{"name": "dal", "quantity": 1, "unit": "bowl", "confidence": 0.9}],
            "note": "dim",
            "error": None,
        }
    )
    assert "Photo analysis" in text and "dal" in text and "confidence 0.90" in text

    not_food = build_system_prompt(vision_result={"is_food": False, "note": "a dog", "items": [], "error": None})
    assert "not food" in not_food


def test_after_ingest_routing():
    assert _after_ingest({"input_type": "text"}) == ["load_context"]
    assert set(_after_ingest({"input_type": "image"})) == {"load_context", "vision_extract"}
    assert set(_after_ingest({"input_type": "image+text"})) == {"load_context", "vision_extract"}


def test_vision_node_noop_without_image():
    assert _vision_extract({"image_path": None}) == {}
