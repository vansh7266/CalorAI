"""Reply-cleaning tests - the output must be plain messaging text."""

from __future__ import annotations

from calorai.agent.formatting import clean_reply, strip_markdown_chunk


def test_strips_bold_and_italic():
    assert clean_reply("Logged **2 rotis** for ~*210* cal") == "Logged 2 rotis for ~210 cal"
    assert clean_reply("__done__") == "done"


def test_strips_headings_bullets_numbers_quotes_backticks():
    md = "# Today\n- roti\n- dal\n1. rice\n> note\nuse `get_meals`"
    out = clean_reply(md)
    assert "#" not in out and "`" not in out
    assert out.startswith("Today")
    assert "- roti" not in out and "1. rice" not in out


def test_leaves_plain_text_alone():
    plain = "You're at 630 cal and 14.5 g protein today. Nice start!"
    assert clean_reply(plain) == plain


def test_does_not_eat_math_asterisk():
    # a lone * between spaces (e.g. "3 * 100") is not markdown emphasis
    assert clean_reply("that's 3 * 100 = 300") == "that's 3 * 100 = 300"


def test_chunk_strip():
    assert strip_markdown_chunk("**1,026 cal**") == "1,026 cal"
