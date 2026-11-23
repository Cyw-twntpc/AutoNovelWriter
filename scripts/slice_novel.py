"""Slice first N chapters of a novel and save as a dedicated test file."""

import sys
from pathlib import Path

sys.path.insert(0, "src")
from autonovel.reference.file_parser import parse_novel_file
from autonovel.reference.macro_extractor import extract_table_of_contents

def slice_novel(input_file: str, output_file: str, max_chapters: int = 305):
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    raw_text, enc = parse_novel_file(input_path)
    print(f"Detected encoding: {enc}, total chars: {len(raw_text)}")

    toc = extract_table_of_contents(raw_text)
    print(f"Total chapters found: {len(toc)}")

    selected = toc[:max_chapters]
    print(f"Slicing first {len(selected)} chapters (1 to {selected[-1]['index']})...")

    # Combine chapters with clean separators
    content = "\n\n".join(f"{ch['title']}\n{ch['text']}" for ch in selected)
    output_path.write_text(content, encoding="utf-8")
    print(f"Saved {len(content)} characters to {output_path}")

    # Verify extracted TOC from the sliced file
    ver_text, ver_enc = parse_novel_file(output_path)
    ver_toc = extract_table_of_contents(ver_text)
    print(f"Verification: parsed {len(ver_toc)} chapters from sliced file. First: '{ver_toc[0]['title']}', Last: '{ver_toc[-1]['title']}'")
    assert len(ver_toc) == len(selected), f"Expected {len(selected)} chapters, got {len(ver_toc)}"
    print("Slice verified successfully!")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    slice_novel("大奉打更人.txt", "data/大奉打更人_前305章.txt", max_chapters=305)
