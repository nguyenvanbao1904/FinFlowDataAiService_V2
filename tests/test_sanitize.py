"""Sanitizer must strip emoji + blockquote while PRESERVING markdown structure
(MarkdownUI on iOS will render it)."""
from __future__ import annotations

import pytest

from app.services.chat.utils.vietnamese_text import sanitize_user_facing_message


@pytest.mark.parametrize("emoji", [
    "📊", "🏦", "✅", "⚠️", "🟢", "🔴", "📌", "📈", "💎", "🔮", "✨", "✖", "★",
])
def test_strip_emoji(emoji):
    assert emoji not in sanitize_user_facing_message(f"{emoji} Nội dung")


def test_preserve_heading():
    src = "## Tiêu đề\n### Phụ đề"
    out = sanitize_user_facing_message(src)
    assert "## Tiêu đề" in out
    assert "### Phụ đề" in out


def test_preserve_bold():
    out = sanitize_user_facing_message("**ROE** là 15%")
    assert "**ROE**" in out


def test_preserve_separator():
    out = sanitize_user_facing_message("Phần A\n\n---\n\nPhần B")
    assert "---" in out


def test_preserve_table():
    src = "| A | B |\n|---|---|\n| 1 | 2 |"
    out = sanitize_user_facing_message(src)
    assert "| A | B |" in out
    assert "|---|---|" in out
    assert "| 1 | 2 |" in out


def test_preserve_bullet():
    out = sanitize_user_facing_message("- Mục 1\n- Mục 2")
    assert "- Mục 1" in out
    assert "- Mục 2" in out


def test_strip_blockquote():
    out = sanitize_user_facing_message("> CAGR ~5%/năm")
    assert ">" not in out
    assert "CAGR" in out


def test_real_world_acb_keeps_markdown_strips_emoji():
    sample = (
        "## Định giá ACB\n\n"
        "| Chỉ tiêu | Giá trị |\n"
        "|---|---|\n"
        "| **Giá hiện tại** | **23,500₫** |\n\n"
        "---\n\n"
        "> Verdict: RẺ\n\n"
        "📊 **Tích cực:**\n"
        "- ✅ ROE cao **15%**\n"
    )
    out = sanitize_user_facing_message(sample)
    # No emoji.
    for ch in "📊✅":
        assert ch not in out, f"Leak emoji: {ch}"
    # Markdown preserved.
    assert "## Định giá ACB" in out
    assert "| Chỉ tiêu | Giá trị |" in out
    assert "**Giá hiện tại**" in out
    assert "---" in out
    assert "**Tích cực:**" in out
    assert "**15%**" in out
    # Blockquote stripped, content kept.
    assert "Verdict: RẺ" in out
    assert ">" not in out


def test_snake_case_replaced():
    out = sanitize_user_facing_message("Theo dõi nim_pct giảm")
    assert "nim_pct" not in out
