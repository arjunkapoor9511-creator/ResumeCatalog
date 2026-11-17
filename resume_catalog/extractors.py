"""Extract logical chunks (one bullet point or one paragraph block each)
from resume files. Each returned LineEntry is already a complete chunk --
no further splitting is needed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from docx import Document
from docx.oxml.ns import qn


@dataclass
class LineEntry:
    text: str
    is_bullet: bool


def extract_lines(identifier: str, stream: BinaryIO) -> list[LineEntry]:
    suffix = Path(identifier).suffix.lower()
    if suffix == ".docx":
        return _extract_docx_lines(stream)
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_docx_lines(stream: BinaryIO) -> list[LineEntry]:
    """A bullet that was typed across multiple paragraphs (soft-wrapped,
    or broken up by a stray Enter) ends up as several <w:p> elements in
    the docx XML, but only the first one carries the list-numbering
    property -- the rest look like plain paragraphs. We treat any
    non-bulleted paragraph that directly follows an open bullet as a
    continuation of it (merged in), regardless of whether the bullet's
    text so far ends with a period -- the only signal for "this is a new
    bullet" is the next paragraph's own list formatting, never
    punctuation. A blank paragraph or a standalone non-bulleted paragraph
    (header, title, date line, etc. -- anything not following an open
    bullet) ends the continuation and is not merged into anything.
    """
    document = Document(stream)
    entries: list[LineEntry] = []
    current_text: str | None = None
    in_bullet_continuation = False

    def flush() -> None:
        nonlocal current_text, in_bullet_continuation
        if current_text is not None:
            entries.append(LineEntry(text=current_text, is_bullet=True))
            current_text = None
        in_bullet_continuation = False

    for paragraph in document.paragraphs:
        text = _paragraph_text(paragraph).strip()
        if not text:
            flush()
            continue
        if _is_docx_list_item(paragraph):
            flush()
            current_text = text
            in_bullet_continuation = True
        elif in_bullet_continuation:
            current_text = f"{current_text} {text}"
        else:
            entries.append(LineEntry(text=text, is_bullet=False))
    flush()
    return entries


_TEXT_TAG = qn("w:t")
_TAB_TAG = qn("w:tab")
_BREAK_TAGS = (qn("w:br"), qn("w:cr"))
_DELETED_TAG = qn("w:del")


def _is_inside_deletion(element) -> bool:
    """True if this element is part of a Track Changes deletion (nested
    under <w:del>) -- that content was proposed for removal, so it's
    excluded to match Word's "No Markup" view, which shows the document
    as it would read once every change is accepted.
    """
    node = element.getparent()
    while node is not None:
        if node.tag == _DELETED_TAG:
            return True
        node = node.getparent()
    return False


def _paragraph_text(paragraph) -> str:
    """python-docx's own `paragraph.text` only walks direct-child <w:r>
    runs of the paragraph, but Track Changes insertions wrap their run in
    a <w:ins> element -- a sibling of plain runs, not a run itself -- so
    that text is silently skipped entirely, not just hidden. Walking
    every descendant instead reaches text nested under <w:ins> (treated
    as accepted, per "No Markup") while <w:delText> is naturally excluded
    since it's a different tag from <w:t>; the <w:del> ancestor check
    additionally excludes any <w:tab>/<w:br> that happens to fall inside
    a deletion.
    """
    parts = []
    for element in paragraph._p.iter():
        if _is_inside_deletion(element):
            continue
        if element.tag == _TEXT_TAG:
            parts.append(element.text or "")
        elif element.tag == _TAB_TAG:
            parts.append("\t")
        elif element.tag in _BREAK_TAGS:
            parts.append("\n")
    return "".join(parts)


def _is_docx_list_item(paragraph) -> bool:
    """A docx bullet/numbered list item carries <w:numPr>, but that
    element can live either directly on the paragraph (bullets applied
    via the toolbar button) or on the paragraph's style definition
    (bullets applied via a style like "List Bullet", which is how
    python-docx and many resumes do it) -- so check the paragraph itself
    first, then walk up its style's base-style chain.
    """
    if _has_num_pr(paragraph._p.pPr):
        return True
    style = paragraph.style
    seen_ids = set()
    while style is not None and style.style_id not in seen_ids:
        seen_ids.add(style.style_id)
        if _has_num_pr(style.element.find(qn("w:pPr"))):
            return True
        style = style.base_style
    return False


def _has_num_pr(pPr) -> bool:
    return pPr is not None and pPr.find(qn("w:numPr")) is not None
