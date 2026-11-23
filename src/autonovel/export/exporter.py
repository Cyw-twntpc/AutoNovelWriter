"""Full-series novel exporter (.md, .txt, .html, .epub, .docx) with range and volume modes."""

from __future__ import annotations

import html
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autonovel.db.project_repo import ProjectRepository
from autonovel.schemas.domain import ExportRequest


class NovelExporter:
    """Exports completed chapters in MD, TXT, HTML, EPUB, and DOCX formats."""

    def __init__(self, repo: ProjectRepository, export_dir: Path) -> None:
        """Initialize exporter with project repository and output folder."""
        self.repo = repo
        self.export_dir = export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_novel(self, req: ExportRequest) -> dict[str, Any]:
        """Filter chapters by requested scope and export to all selected formats."""
        cfg = self.repo.get_project_config()
        novel_title = str(cfg.get("title") or "未命名長篇小說")
        chapters = self._query_chapters_for_scope(req)
        if not chapters:
            raise ValueError("選定範圍內尚無已完成之章節可供匯出。")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        # Two exports in the same second used to share a folder and silently overwrite
        # each other; microsecond stamps plus an explicit collision loop prevent that.
        batch_dir = self.export_dir / f"export_{ts}"
        suffix = 1
        while True:
            try:
                batch_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                batch_dir = self.export_dir / f"export_{ts}_{suffix}"
                suffix += 1
        generated_files: list[str] = []
        if req.merge_mode == "PER_VOLUME":
            generated_files.extend(
                self._export_per_volume(batch_dir, novel_title, chapters, req.formats)
            )
        else:
            generated_files.extend(
                self._export_group(batch_dir, novel_title, chapters, req.formats)
            )
        return {
            "export_dir": str(batch_dir),
            "chapter_count": len(chapters),
            "files": generated_files,
        }

    def _query_chapters_for_scope(self, req: ExportRequest) -> list[dict[str, Any]]:
        """Query completed/revised chapters matching ALL, VOLUME, or RANGE scope."""
        base_sql = (
            "SELECT chapter_num, volume_num, title, final_content "
            "FROM chapters WHERE status IN ('COMPLETED', 'REVISED') "
        )
        scope = (req.scope_type or "ALL").strip().upper()
        if scope == "VOLUME":
            if req.volume_num is None:
                # Silently falling back to the whole book would export far more than
                # the user asked for.
                raise ValueError("匯出範圍為「指定卷」時必須提供 volume_num。")
            return self.repo.fetch_all(
                base_sql + "AND volume_num = ? ORDER BY chapter_num ASC;",
                (req.volume_num,),
            )
        if scope == "RANGE":
            if req.start_chapter is None or req.end_chapter is None:
                raise ValueError(
                    "匯出範圍為「章節區間」時必須同時提供 start_chapter 與 end_chapter。"
                )
            if req.start_chapter > req.end_chapter:
                raise ValueError("起始章節不可大於結束章節。")
            return self.repo.fetch_all(
                base_sql + "AND chapter_num BETWEEN ? AND ? ORDER BY chapter_num ASC;",
                (req.start_chapter, req.end_chapter),
            )
        if scope != "ALL":
            raise ValueError(f"不支援的匯出範圍類型：{req.scope_type}")
        return self.repo.fetch_all(base_sql + "ORDER BY chapter_num ASC;")

    def _export_per_volume(
        self,
        batch_dir: Path,
        novel_title: str,
        chapters: list[dict[str, Any]],
        formats: list[str],
    ) -> list[str]:
        """Group chapters by volume_num into separate subdirectories and export each."""
        by_vol: dict[int, list[dict[str, Any]]] = {}
        for ch in chapters:
            by_vol.setdefault(int(ch["volume_num"]), []).append(ch)
        created: list[str] = []
        for vol_num, vol_chapters in sorted(by_vol.items()):
            vol_folder = batch_dir / f"Volume_{vol_num:02d}"
            vol_folder.mkdir(parents=True, exist_ok=True)
            vol_title = f"{novel_title}_第{vol_num}卷"
            created.extend(
                self._export_group(vol_folder, vol_title, vol_chapters, formats)
            )
        return created

    def _export_group(
        self,
        target_dir: Path,
        book_title: str,
        chapters: list[dict[str, Any]],
        formats: list[str],
    ) -> list[str]:
        """Write a chapter list into all requested formats inside target_dir."""
        safe_name = "".join(
            c for c in book_title if c.isalnum() or c in ("_", "-", "《", "》", "·")
        ).strip("《》") or "novel"
        out_files: list[str] = []
        for fmt in formats:
            ext = fmt.lower().lstrip(".")
            dest = target_dir / f"{safe_name}.{ext}"
            if ext == "md":
                self._write_markdown(dest, book_title, chapters)
            elif ext == "txt":
                self._write_txt(dest, book_title, chapters)
            elif ext == "html":
                self._write_html(dest, book_title, chapters)
            elif ext == "epub":
                self._write_epub(dest, book_title, chapters)
            elif ext == "docx":
                self._write_docx(dest, book_title, chapters)
            else:
                continue
            out_files.append(str(dest))
        return out_files

    @staticmethod
    def _write_markdown(
        dest: Path, title: str, chapters: list[dict[str, Any]]
    ) -> None:
        """Write Markdown (.md) file with book header and chapter sections."""
        blocks = [f"# {title}\n"]
        for ch in chapters:
            blocks.append(f"## {ch['title']}\n\n{ch['final_content'].strip()}\n")
        dest.write_text("\n".join(blocks), encoding="utf-8")

    @staticmethod
    def _write_txt(dest: Path, title: str, chapters: list[dict[str, Any]]) -> None:
        """Write plain text (.txt) file with standard Chinese full-width indentation."""
        blocks = [f"{title}\n{'=' * 40}\n"]
        for ch in chapters:
            paras = [
                f"　　{p.strip()}"
                for p in str(ch["final_content"]).splitlines()
                if p.strip()
            ]
            blocks.append(f"\n{ch['title']}\n\n" + "\n\n".join(paras) + "\n")
        dest.write_text("\n".join(blocks), encoding="utf-8")

    @staticmethod
    def _write_html(dest: Path, title: str, chapters: list[dict[str, Any]]) -> None:
        """Write standalone styled HTML (.html) reader package with TOC."""
        toc_items = [
            f"<li><a href='#ch-{c['chapter_num']}'>{html.escape(str(c['title']))}</a></li>"
            for c in chapters
        ]
        ch_sections: list[str] = []
        for c in chapters:
            paras = "".join(
                f"<p>{html.escape(p.strip())}</p>"
                for p in str(c["final_content"]).splitlines()
                if p.strip()
            )
            ch_sections.append(
                f"<section id='ch-{c['chapter_num']}'><h2>{html.escape(str(c['title']))}</h2>{paras}</section>"
            )
        doc = (
            "<!DOCTYPE html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>"
            "<style>body{max-width:820px;margin:2rem auto;font-family:'Noto Serif TC',serif;"
            "background:#faf8f5;color:#222;line-height:1.9;padding:0 1.5rem;}"
            "p{text-indent:2em;margin:0.8em 0;}section{margin-top:3rem;border-top:1px solid #ddd;padding-top:1.5rem;}</style>"
            f"</head><body><h1>{html.escape(title)}</h1><nav><h3>目錄</h3><ul>{''.join(toc_items)}</ul></nav>"
            f"{''.join(ch_sections)}</body></html>"
        )
        dest.write_text(doc, encoding="utf-8")

    @staticmethod
    def _write_epub(dest: Path, title: str, chapters: list[dict[str, Any]]) -> None:
        """Build a valid EPUB 3 archive with cover page, navigation TOC, and XHTML chapters."""
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED
            )
            zf.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>'
                "</container>",
            )
            manifest_items = [
                '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
            ]
            spine_items: list[str] = []
            nav_links: list[str] = []
            for ch in chapters:
                cnum = int(ch["chapter_num"])
                fname = f"chapter_{cnum:04d}.xhtml"
                manifest_items.append(
                    f'<item id="ch{cnum}" href="{fname}" media-type="application/xhtml+xml"/>'
                )
                spine_items.append(f'<itemref idref="ch{cnum}"/>')
                nav_links.append(
                    f'<li><a href="{fname}">{html.escape(str(ch["title"]))}</a></li>'
                )
                paras = "".join(
                    f"<p>{html.escape(p.strip())}</p>"
                    for p in str(ch["final_content"]).splitlines()
                    if p.strip()
                )
                xhtml = (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="zh-Hant">'
                    f"<head><title>{html.escape(str(ch['title']))}</title></head>"
                    f"<body><h2>{html.escape(str(ch['title']))}</h2>{paras}</body></html>"
                )
                zf.writestr(f"OEBPS/{fname}", xhtml)
            nav_doc = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
                f"<head><title>{html.escape(title)}</title></head>"
                f'<body><nav epub:type="toc"><h1>{html.escape(title)}</h1><ol>{"".join(nav_links)}</ol></nav></body></html>'
            )
            zf.writestr("OEBPS/nav.xhtml", nav_doc)
            # Derive the identifier from the book title plus its chapter span so two
            # different books with the same chapter count stop sharing a UUID.
            chapter_key = ",".join(str(int(c["chapter_num"])) for c in chapters)
            book_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"autonovel:{title}:{chapter_key}")
            modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            opf = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">'
                '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                f"<dc:identifier id=\"bookid\">urn:uuid:{book_uuid}</dc:identifier>"
                f"<dc:title>{html.escape(title)}</dc:title><dc:language>zh-TW</dc:language>"
                f'<meta property="dcterms:modified">{modified}</meta></metadata>'
                f"<manifest>{''.join(manifest_items)}</manifest>"
                f"<spine>{''.join(spine_items)}</spine></package>"
            )
            zf.writestr("OEBPS/content.opf", opf)

    @staticmethod
    def _write_docx(dest: Path, title: str, chapters: list[dict[str, Any]]) -> None:
        """Build a valid OpenXML Word (.docx) package with book title and chapter headings."""
        w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        body_xml = [
            f"<w:p><w:r><w:rPr><w:b/><w:sz w:val='44'/></w:rPr><w:t>{html.escape(title)}</w:t></w:r></w:p>"
        ]
        for ch in chapters:
            body_xml.append(
                f"<w:p><w:r><w:rPr><w:b/><w:sz w:val='32'/></w:rPr><w:t>{html.escape(str(ch['title']))}</w:t></w:r></w:p>"
            )
            for p in str(ch["final_content"]).splitlines():
                if p.strip():
                    body_xml.append(
                        f"<w:p><w:r><w:t>　　{html.escape(p.strip())}</w:t></w:r></w:p>"
                    )
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{w_ns}"><w:body>{"".join(body_xml)}<w:sectPr/></w:body></w:document>'
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        )
        rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>"
        )
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", rels)
            zf.writestr("word/document.xml", doc_xml)
