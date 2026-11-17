from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from resume_catalog.catalog_builder import CatalogBuilder
from resume_catalog.chunker import split_into_chunks
from resume_catalog.extractors import extract_lines
from resume_catalog.sources import DEFAULT_EXTENSIONS, LocalFolderSource

logger = logging.getLogger(__name__)


def _compile_regex(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"invalid regex {pattern!r}: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract bullet points/paragraphs from .docx resume files in a folder "
        "(other file types are ignored), dedup them, and write a JSON catalog."
    )
    parser.add_argument("--input", required=True, help="Folder containing .docx resume files")
    parser.add_argument("--output", default="catalog.json", help="Path to write the catalog JSON")
    parser.add_argument(
        "--name-pattern",
        action="append",
        type=_compile_regex,
        metavar="REGEX",
        dest="name_patterns",
        help="Only include files whose filename matches this regex (case-insensitive, "
        "matched anywhere in the name, e.g. 'resume' or 'cv'). Repeatable -- a file is "
        "included if it matches ANY given pattern. If omitted, every file is included.",
    )
    parser.add_argument(
        "--bullets-only",
        action="store_true",
        help="Only write bullet-point chunks to the catalog; skip plain paragraph chunks.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every new chunk found")
    return parser


def run(
    input_folder: str,
    output_path: str,
    name_patterns: list[re.Pattern[str]] | None = None,
    bullets_only: bool = False,
) -> None:
    source = LocalFolderSource(input_folder, extensions=DEFAULT_EXTENSIONS)
    files = source.list_files()
    if name_patterns:
        found = len(files)
        files = [f for f in files if any(p.search(Path(f).name) for p in name_patterns)]
        logger.info(
            "Found %d file(s) in %s, %d matched --name-pattern", found, input_folder, len(files)
        )
    else:
        logger.info("Found %d file(s) in %s", len(files), input_folder)

    builder = CatalogBuilder()
    total_chunks = 0
    for identifier in files:
        logger.info("Processing %s", identifier)
        with source.open(identifier) as stream:
            lines = extract_lines(identifier, stream)
        chunks = split_into_chunks(lines)
        if bullets_only:
            chunks = [chunk for chunk in chunks if chunk.is_bullet]
        total_chunks += len(chunks)
        for chunk in chunks:
            builder.add(chunk.text, source=identifier, is_bullet=chunk.is_bullet)

    catalog = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_folder": input_folder,
        "name_patterns": [p.pattern for p in name_patterns] if name_patterns else None,
        "bullets_only": bullets_only,
        "total_files": len(files),
        "total_chunks_extracted": total_chunks,
        "unique_chunks": len(builder),
        "entries": builder.to_list(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    logger.info(
        "Wrote %d unique chunk(s) out of %d extracted to %s",
        len(builder),
        total_chunks,
        output_path,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    run(args.input, args.output, name_patterns=args.name_patterns, bullets_only=args.bullets_only)


if __name__ == "__main__":
    main()
