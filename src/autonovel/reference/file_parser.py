"""Multi-format file parser (.txt/.md/.epub/.docx) and 1-second SHA-256 pre-flight check."""

from __future__ import annotations

import hashlib
import html.entities as html_entities
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote

from autonovel.db.reference_repo import ReferenceRepository
from autonovel.schemas.domain import PreflightFileItem

SUPPORTED_EXTENSIONS = {".txt", ".md", ".epub", ".docx"}
CANDIDATE_ENCODINGS = ("utf-8", "utf-8-sig", "big5-hkscs", "big5", "gb18030", "gbk")

# High-frequency Han characters (simplified + traditional). Real prose hits many of
# them, while a wrong-charset decode (e.g. GBK bytes read as Big5) produces near-zero
# matches, which lets us pick the right encoding instead of trusting list order.
_COMMON_HANZI = set(
    "的一是不了在人有我他她它這这中大來来上國国個个到說说們们為为子和你地出道也時时年得就"
    "那要下以生會会自著着去之過过家學学對对可她裡里後后小麼么心多天而能好都然沒没日於于起"
    "還还發发成事只作想當当看文無无開开手十用主行方又如前所本見见經经頭头面公同三已老從从"
    "動动兩两長长知民樣样現现分將将外但身些與与高意進进把法此實实回二理美點点月明其種种聲"
    "声全工己話话兒儿者向情部正名定女問问力機机給给等幾几很業业最間间新什打便位因重被走電"
    "电四第門门相次東东政海口使教西再平真聽听世氣气信北少關关並并內内加化由卻却代軍军產产"
    "入先山五太水萬万市眼體体別别處处總总才場场師师書书比住員员九笑性通目華华報报馬马命張"
    "张活難难神數数件安表原車车白應应路期叫死常提感金何更反合放做系計计或司利受光王果親亲"
    "界及今京務务制解各任至清物台象記记邊边共風风戰战干接許许八特覺觉望直服毛林題题建南度"
    "統统色字請请交愛爱讓让認认算論论百吃義义科怎元社術术結结六功指思非流每青管夫連连遠远"
    "資资隊队跟帶带花快條条院變变聯联言權权往展該该領领傳传近留紅红治決决周保達达辦办運运"
    "武半候七必城父強强步完革深區区即求品士轉转量空甚眾众技輕轻程告江語语英基派滿满式李息"
    "寫写呢識识極极令黃黄德收臉脸錢钱黨党倒未持取設设始版雙双歷历越史商千片容研像找友孩站"
    "廣广改議议形委早房音火際际則则首單单據据導导影失拿網网香似斯專专石若兵弟誰谁校讀读志"
    "飛飞觀观爭争究包組组造落視视濟济喜離离雖虽坐集編编寶宝談谈府拉黑且隨随格盡尽劍剑講讲"
    "布殺杀微怕母調调局根曾準准段終终樂乐苦創创刻靠夠够誤误歡欢醒執执筆笔紙纸"
)

_NAMED_ENTITY_RE = re.compile(r"&([A-Za-z][A-Za-z0-9]{1,31});")
_DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>]*(\[[^\]]*\])?[^>]*>", re.IGNORECASE | re.DOTALL)
_XML_BUILTIN_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}


def _xml_from_markup(raw: bytes) -> ET.Element:
    """Parse XHTML/XML tolerantly: drop DOCTYPE and resolve HTML named entities.

    XHTML books commonly use ``&nbsp;``/``&mdash;`` without declaring them, which
    makes ElementTree raise ParseError and the book look corrupted.
    """
    text = raw.decode("utf-8", errors="replace")
    text = _DOCTYPE_RE.sub("", text)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in _XML_BUILTIN_ENTITIES:
            return match.group(0)
        return html_entities.html5.get(name + ";", " ")

    text = _NAMED_ENTITY_RE.sub(_replace, text)
    return ET.fromstring(text)


def _local_name(tag: str) -> str:
    """Return the tag name without any XML namespace prefix."""
    return tag.rsplit("}", 1)[-1]


def _epub_spine_paths(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    """Resolve OPF manifest/spine into ordered XHTML paths; empty when unavailable."""
    try:
        container = _xml_from_markup(zf.read("META-INF/container.xml"))
    except (KeyError, ET.ParseError):
        return []
    rootfile = next(
        (e for e in container.iter() if _local_name(e.tag) == "rootfile"), None
    )
    opf_path = (rootfile.get("full-path", "") if rootfile is not None else "").strip()
    if not opf_path or opf_path not in names:
        return []
    try:
        opf = _xml_from_markup(zf.read(opf_path))
    except (KeyError, ET.ParseError):
        return []
    manifest: dict[str, str] = {}
    for item in opf.iter():
        if _local_name(item.tag) == "item":
            item_id, href = item.get("id"), item.get("href")
            if item_id and href:
                manifest[item_id] = href
    base = posixpath.dirname(opf_path)
    ordered: list[str] = []
    for ref in opf.iter():
        if _local_name(ref.tag) != "itemref":
            continue
        href = manifest.get(ref.get("idref", ""))
        if not href:
            continue
        target = unquote(href.split("#", 1)[0])
        ordered.append(posixpath.normpath(posixpath.join(base, target)))
    return ordered


def compute_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_novel_file(file_path: Path) -> tuple[str, str]:
    """Parse text from .txt, .md, .epub, or .docx and return (text, encoding_label)."""
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")
    if ext in (".txt", ".md"):
        return _read_plain_text_with_encoding(file_path)
    if ext == ".docx":
        return _read_docx_text(file_path), "ooxml-utf8"
    return _read_epub_text(file_path), "epub-utf8"


def _read_plain_text_with_encoding(file_path: Path) -> tuple[str, str]:
    """Attempt decoding raw bytes across UTF-8, GB18030/GBK, Big5, and UTF-16."""
    raw = file_path.read_bytes()

    # 1. Check for UTF-16 BOM signatures (standard Windows Notepad Unicode)
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                text = raw.decode(enc)
                if text.strip():
                    return text, enc
            except UnicodeDecodeError:
                continue

    # 2. Try standard candidate encodings if no null bytes in header.
    # Score every successful decode instead of returning the first one: Big5 precedes
    # GB18030 in the candidate list, so order alone mis-decodes GBK books into mojibake.
    if b"\x00" not in raw[:1024]:
        if raw.isascii():
            return raw.decode("ascii"), "ascii"
        best: tuple[float, str, str] | None = None
        for enc in CANDIDATE_ENCODINGS:
            try:
                text = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if not text.strip():
                continue
            # Reject false positive decodes (e.g. GB18030 mapping Big5 bytes to Private Use Area characters)
            sample = text[:4000]
            pua_count = sum(1 for c in sample if 0xE000 <= ord(c) <= 0xF8FF)
            if pua_count / max(1, len(sample)) > 0.03:
                continue
            hits = sum(1 for c in sample if c in _COMMON_HANZI)
            score = hits / max(1, len(sample))
            if best is None or score > best[0]:
                best = (score, text, enc)
        if best is not None:
            return best[1], best[2]

    # 3. If null bytes exist, test for UTF-16 LE / BE without BOM before rejecting
    for enc in ("utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = raw.decode(enc)
            # Ensure text decoded cleanly without interior null chars
            if text.strip() and "\x00" not in text[:512]:
                return text, enc
        except UnicodeDecodeError:
            continue

    if b"\x00" in raw[:1024]:
        raise ValueError("Binary null bytes detected in text file.")
    raise ValueError("Unable to decode text file with UTF-8, GBK, Big5, or UTF-16.")


def _read_docx_text(file_path: Path) -> str:
    """Extract paragraph text from a .docx ZIP archive (word/document.xml)."""
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            xml_bytes = zf.read("word/document.xml")
        root = _xml_from_markup(xml_bytes)
        paragraphs: list[str] = []
        for elem in root.iter():
            if elem.tag.endswith("}p"):
                texts = [t.text for t in elem.iter() if t.tag.endswith("}t") and t.text]
                line = "".join(texts).strip()
                if line:
                    paragraphs.append(line)
        content = "\n".join(paragraphs)
        if not content.strip():
            raise ValueError("Empty .docx content.")
        return content
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise ValueError(f"Corrupted .docx file: {exc}") from exc


def _read_epub_text(file_path: Path) -> str:
    """Extract readable text from XHTML/HTML items inside an .epub archive."""
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            html_names = [
                n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))
            ]
            if not html_names:
                raise ValueError("No XHTML chapters found in .epub archive.")
            available = set(zf.namelist())
            # Follow the OPF spine so chapter order matches the book; alphabetical
            # sorting would put chapter10 before chapter2.
            ordered = [
                n for n in _epub_spine_paths(zf, available) if n in available
            ]
            if not ordered:
                ordered = sorted(html_names)
            sections: list[str] = []
            for name in ordered:
                raw_html = zf.read(name)
                root = _xml_from_markup(raw_html)
                text = "".join(root.itertext()).strip()
                if text:
                    sections.append(text)
            content = "\n\n".join(sections)
            if not content.strip():
                raise ValueError("Empty .epub content.")
            return content
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError(f"Corrupted .epub archive: {exc}") from exc


def preflight_check_files(
    file_paths: list[Path], repo: ReferenceRepository
) -> list[PreflightFileItem]:
    """Run static pre-flight check for duplicates, encoding, and file corruption."""
    results: list[PreflightFileItem] = []
    for path in file_paths:
        if not path.exists() or not path.is_file():
            results.append(
                PreflightFileItem(
                    file_path=str(path),
                    file_name=path.name,
                    file_sha256="",
                    encoding="",
                    status="CORRUPTED",
                    error_reason="File does not exist.",
                )
            )
            continue
        try:
            sha = compute_sha256(path)
            _, encoding = parse_novel_file(path)
            existing_src = repo.get_source_by_hash(sha)
            is_dup = existing_src is not None
            results.append(
                PreflightFileItem(
                    file_path=str(path),
                    file_name=path.name,
                    file_sha256=sha,
                    encoding=encoding,
                    status="DUPLICATE" if is_dup else "NEW",
                    has_skills=bool(existing_src.get("has_skills", False)) if existing_src else False,
                    entry_count=int(existing_src.get("entry_count", 0)) if existing_src else 0,
                    has_pacing=bool(existing_src.get("has_pacing", False)) if existing_src else False,
                    has_blueprint=bool(existing_src.get("has_blueprint", False)) if existing_src else False,
                )
            )
        except Exception as exc:
            results.append(
                PreflightFileItem(
                    file_path=str(path),
                    file_name=path.name,
                    file_sha256="",
                    encoding="",
                    status="CORRUPTED",
                    error_reason=str(exc),
                )
            )
    return results
