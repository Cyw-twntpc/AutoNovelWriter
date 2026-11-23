"""Prose normalization engine: Simplified-to-Traditional conversion, quote normalization, and web noise filtering."""

from __future__ import annotations

import json
from pathlib import Path
import re

# Regex matching external web-novel file impurities (patron requests, author notes, piracy notices)
WEB_NOISE_RE = re.compile(
    r"(?:"
    r"^\s*(?:PS|ps|附|作者有話說|作者有话说|作者ps)[：:][^\n]*$|"
    r"^\s*.*(?:求月票|求推薦票|求推荐票|求收藏|求追讀|求追读|求訂閱|求订阅|打賞感謝|打赏感谢|感謝盟主|感谢盟主|投月票|投推薦票|投推荐票).*$|"
    r"^\s*.*(?:本書首發|本书首发|請訪問|请访问|筆趣閣|笔趣阁|最新章節|最新章节|轉載請註明|转载请注明|手機用戶請瀏覽|手机用户请浏览).*$"
    r")",
    re.MULTILINE | re.IGNORECASE,
)

# Quotation mark pairs normalization for Traditional Chinese fiction
QUOTE_PAIRS = [
    ("“", "「"),
    ("”", "」"),
    ("‘", "『"),
    ("’", "』"),
]

# Literary vocabulary normalization for Taiwanese/Traditional standard
LITERARY_TERM_MAP: dict[str, str] = {
    "千裏": "千里",
    "萬裏": "萬里",
    "百裏": "百里",
    "飄着": "飄著",
    "隨着": "隨著",
    "跟着": "跟著",
    "接着": "接著",
    "站着": "站著",
    "看著": "看著",
    "凝視着": "凝視著",
    "拿着": "拿著",
    "握着": "握著",
    "等着": "等著",
    "朝着": "朝著",
    "防御": "防禦",
    "尽快": "儘快",
    "作爲": "作為",
    "修爲": "修為",
    "那裏": "那裡",
    "這裏": "這裡",
    "此章节": "此章節",
    "下章将": "下章將",
    "門派": "門派",
    "修煉": "修煉",
    "天劫": "天劫",
}

# Load OpenCC 4000+ characters table
_DATA_FILE = Path(__file__).parent / "opencc_data.json"
if _DATA_FILE.is_file():
    try:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            _raw_data = json.load(f)
            _S_STR = _raw_data.get("s", "")
            _T_STR = _raw_data.get("t", "")
            _S2T_TRANS = str.maketrans(_S_STR, _T_STR)
    except Exception:
        _S2T_TRANS = None
else:
    _S2T_TRANS = None


def convert_simplified_to_traditional(text: str) -> str:
    """Fast deterministic conversion from Simplified Chinese to Traditional Chinese."""
    if not text:
        return ""
    if _S2T_TRANS:
        translated = text.translate(_S2T_TRANS)
    else:
        translated = text
    for s_term, t_term in LITERARY_TERM_MAP.items():
        translated = translated.replace(s_term, t_term)
    return translated


def normalize_quotes_to_traditional(text: str) -> str:
    """Convert western/mainland quotation marks to traditional Chinese angle brackets."""
    res = text
    for w_quote, t_quote in QUOTE_PAIRS:
        res = res.replace(w_quote, t_quote)
    return res


def strip_web_novel_noise(text: str) -> str:
    """Filter out patron solicitations, author chatter, and piracy watermark lines from raw imported files."""
    if not text:
        return ""
    lines = text.splitlines()
    cleaned = [ln for ln in lines if not WEB_NOISE_RE.match(ln)]
    return "\n".join(cleaned)


def sanitize_novel_prose(text: str) -> str:
    """Normalize character encoding and punctuation into Traditional Chinese without deleting or mutating AI content.

    1. Converts simplified characters to traditional Chinese.
    2. Normalizes dialogue quotes to traditional corner brackets.
    3. Collapses redundant blank lines.
    """
    if not text:
        return ""
    stage1 = convert_simplified_to_traditional(text)
    stage2 = normalize_quotes_to_traditional(stage1)
    lines = [p.rstrip() for p in stage2.splitlines()]
    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1 and collapsed:
                collapsed.append("")
        else:
            blank_run = 0
            collapsed.append(line)
    return "\n".join(collapsed).strip()