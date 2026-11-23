"""Unit tests for non-destructive prose normalization engine."""

from autonovel.engine.prose_sanitizer import (
    convert_simplified_to_traditional,
    normalize_quotes_to_traditional,
    sanitize_novel_prose,
)

SAMPLE_RAW_CHAPTER = """
寒淵宗位於千里之外的幽冥之地，四周被无尽的寒冰覆盖。晨雾弥漫，山间飘着一股阴冷的寒气。


李逍遥的身边站着他的师兄弟们，三人皆是寒淵宗的顶尖弟子，修炼天赋极佳。
李逍遥轻声说道：“师兄、师姐，最近宗门中气氛似乎有些不对劲，你们感觉到了吗？”
""".strip()


def test_sanitize_novel_prose_preserves_content_and_normalizes():
    """Verify sanitize_novel_prose converts simplified to traditional and normalizes quotes without deleting text."""
    cleaned = sanitize_novel_prose(SAMPLE_RAW_CHAPTER)
    # Simplified converted to Traditional
    assert "无尽" not in cleaned
    assert "無盡" in cleaned
    assert "覆盖" not in cleaned
    assert "覆蓋" in cleaned
    assert "晨雾" not in cleaned
    assert "晨霧" in cleaned
    assert "飘着" not in cleaned
    assert "飄著" in cleaned
    assert "师兄弟们" not in cleaned
    assert "師兄弟們" in cleaned
    # Dialogue quotes normalized to Traditional corner brackets
    assert "「" in cleaned
    assert "」" in cleaned
    assert "“" not in cleaned
    assert "”" not in cleaned
    # Excessive blank lines collapsed to single blank line
    assert "\n\n\n" not in cleaned
    # All prose content preserved
    assert "寒淵宗位於千里之外的幽冥之地" in cleaned
    assert "最近宗門中氣氛似乎有些不對勁" in cleaned


def test_quote_normalization_nested():
    """Verify curly quotes normalize to traditional corner brackets."""
    raw = "他沉聲道：“記住‘天道無親’這四個字。”"
    norm = normalize_quotes_to_traditional(raw)
    assert norm == "他沉聲道：「記住『天道無親』這四個字。」"


def test_simplified_to_traditional():
    """Verify OpenCC simplified to traditional conversion."""
    assert convert_simplified_to_traditional("风云激荡") == "風雲激蕩"
