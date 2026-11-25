"""Parse a resume (.docx) into structured JSON using Claude Haiku.

Usage:
    python parse_resume.py path/to/resume.docx [-o out.json | -d output_folder]

Structure is enforced with Anthropic structured outputs: `messages.parse`
sends the Pydantic schema below as `output_config.format`, the API constrains
decoding to that JSON schema, and the SDK validates the result back into a
`ParsedResume` instance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pydantic import BaseModel, Field

MODEL = "claude-haiku-5-5"

# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------

DATE = "YYYY-MM when the month is known, else YYYY; 'Present' for current; empty string if absent."


class Bullet(BaseModel):
    text: str = Field(description="The full bullet exactly as written in the resume, verbatim.")


class Role(BaseModel):
    title: str
    start_date: str = Field(description=DATE)
    end_date: str = Field(description=DATE)
    bullets: list[Bullet] = Field(
        description="Bullets for this position. Bullets that don't clearly belong to one "
        "position go under the most recently held role at the company."
    )


class Company(BaseModel):
    name: str = Field(description="Company name in normal casing, without parenthetical tags.")
    location: str
    start_date: str = Field(description="Earliest role start at this company. " + DATE)
    end_date: str = Field(description="Latest role end at this company. " + DATE)
    company_summary: str = Field(
        description="The non-bulleted sentence describing what the company is, verbatim. "
        "Exclude parenthetical tags beside the company name (e.g. '(AI Startup)', '(NASDAQ: MSFT)') "
        "and add no words or punctuation of your own. Empty string if there is no such sentence."
    )
    role_summary: str = Field(
        description="The non-bulleted line describing the candidate's overall role or remit at this company, verbatim."
    )
    roles: list[Role] = Field(description="Positions held at this company, most recent first.")


class Education(BaseModel):
    institution: str
    location: str
    degree: str = Field(
        description="Degree type only, e.g. 'Bachelor of Science', 'Master of Business Administration'. "
        "Split 'X in Y' into degree X and field_of_study Y. Honours/awards go in notes, not here."
    )
    field_of_study: str = Field(description="e.g. 'Computing & Software Systems'. Empty string if not stated.")
    graduation_date: str
    notes: list[str] = Field(
        description="Bullets under this education entry and honours/awards attached to the degree "
        "(e.g. \"(Dean's Award)\"), verbatim."
    )


class AdditionalItem(BaseModel):
    title: str
    organization: str
    location: str
    start_date: str = Field(description=DATE)
    end_date: str = Field(description=DATE)
    text: str = Field(
        description="The entire entry exactly as written, verbatim, including any heading line and "
        "bullets (one per line)."
    )


class Contact(BaseModel):
    name: str
    email: str
    phone: str
    location: str
    links: list[str] = Field(
        description="Other contact-section items such as profile links or link text, exactly as they appear."
    )


class ParsedResume(BaseModel):
    contact: Contact
    headline: str = Field(description="Title line under the name, if any.")
    summary: str = Field(description="Professional summary paragraph, verbatim, if any.")
    companies: list[Company] = Field(
        description="Everything the resume lists under its work-experience section, grouped by "
        "employer, in resume order."
    )
    education: list[Education]
    skills: list[str]
    certifications: list[str]
    additional_information: list[AdditionalItem] = Field(
        description="Entries the resume lists outside work experience and education: volunteer work, "
        "side projects, teaching, advisory roles, interests, 'Additional' section entries."
    )
    unclassified: list[str] = Field(
        description="Any resume text that fits nowhere else, verbatim. Nothing may be dropped."
    )


# --------------------------------------------------------------------------
# .docx -> text
# --------------------------------------------------------------------------


def _is_list_item(paragraph: Paragraph) -> bool:
    ppr = paragraph._p.pPr
    if ppr is not None and ppr.find(qn("w:numPr")) is not None:
        return True
    style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
    return "list" in style or "bullet" in style


def _paragraph_line(paragraph: Paragraph) -> str | None:
    # Visible text only (hyperlink text included, link targets ignored).
    text = paragraph.text.strip()
    if not text:
        return None
    return f"• {text}" if _is_list_item(paragraph) else text


def docx_to_text(path: Path) -> str:
    """Walk the document body in order, keeping tables (common in resume
    layouts) and marking list items with '•' so the model can tell bullets
    from headers."""
    document = Document(str(path))
    lines: list[str] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            line = _paragraph_line(Paragraph(child, document))
            if line:
                lines.append(line)
        elif child.tag == qn("w:tbl"):
            for row in Table(child, document).rows:
                seen: set[int] = set()
                for cell in row.cells:
                    if id(cell._tc) in seen:  # merged cells repeat
                        continue
                    seen.add(id(cell._tc))
                    for p in cell.paragraphs:
                        line = _paragraph_line(p)
                        if line:
                            lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Claude call
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You extract resumes into structured data for a story bank.

Rules:
- Copy bullets, summaries and additional entries verbatim. Do not fix, shorten, or rephrase.
  The exact wording matters: the same story phrased differently across resumes is
  what the downstream system tracks.
- Group experience by company, then by the positions held there. Lines starting with "•"
  are bullets. If a company lists several positions and it isn't clear which one a bullet
  belongs to, put it under the most recently held position.
- Under a company heading, a non-bulleted sentence describing the company itself is
  company_summary; one describing what the candidate did there overall is role_summary.
- company_summary is only the resume's own sentence about the company. Leave out tags like
  "(AI Startup)" next to the company name, and never add joining words or punctuation.
- Follow the resume's own sections: anything listed under work experience goes in companies
  (even teaching or volunteer roles); entries listed elsewhere (volunteer work, side projects,
  teaching, advisory work, interests, an "Additional" section) go in additional_information.
- For each additional_information entry, put the entire entry verbatim in text, including any
  heading line and its bullets.
- Use "" (empty string) for fields the resume does not contain, and [] for empty lists. Never infer facts.
- Every piece of resume content must land somewhere; use unclassified if nothing fits."""


def parse_resume(text: str, client: anthropic.Anthropic) -> ParsedResume:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<resume>\n{text}\n</resume>"}],
        output_format=ParsedResume,
    )

    if response.stop_reason == "refusal":
        details = response.stop_details
        raise RuntimeError(
            f"Model declined the request ({details.category if details else 'no details'})."
        )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Output hit max_tokens before the JSON was complete; raise max_tokens.")
    if response.parsed_output is None:
        raise RuntimeError(f"No parsed output (stop_reason={response.stop_reason}).")
    return response.parsed_output


def blanks_to_null(value):
    """The schema uses "" for missing fields (nullable unions bloat the
    structured-output grammar); restore null for scalar fields on output.
    Empty strings inside lists are dropped."""
    if isinstance(value, dict):
        return {k: (None if v == "" else blanks_to_null(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [blanks_to_null(v) for v in value if v != ""]
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resume", type=Path, help="Path to a .docx resume")
    dest = ap.add_mutually_exclusive_group()
    dest.add_argument("-o", "--output", type=Path, help="Output JSON file path")
    dest.add_argument(
        "-d", "--output-dir", type=Path,
        help="Folder to write <resume name>.json into (created if missing). "
        "Default: next to the resume.",
    )
    ap.add_argument("--dump-text", action="store_true", help="Print the extracted text and exit (no API call)")
    args = ap.parse_args()

    if not args.resume.is_file():
        print(f"File not found: {args.resume}", file=sys.stderr)
        return 2
    if args.output_dir and args.output_dir.exists() and not args.output_dir.is_dir():
        print(f"Output folder is an existing file: {args.output_dir}", file=sys.stderr)
        return 2
    if args.resume.suffix.lower() != ".docx":
        print("Only .docx is supported. Re-save legacy .doc files as .docx.", file=sys.stderr)
        return 2

    text = docx_to_text(args.resume)
    if args.dump_text:
        print(text)
        return 0
    if not text.strip():
        print("No text found in document.", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    try:
        parsed = parse_resume(text, client)
    except anthropic.RateLimitError as e:
        print(f"Rate limited, try again shortly: {e}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as e:
        print(f"Could not reach the API: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    except TypeError as e:
        if "authentication" not in str(e):
            raise
        print(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY in this terminal, e.g.\n"
            '  $env:ANTHROPIC_API_KEY = "sk-ant-..."',
            file=sys.stderr,
        )
        return 1

    if args.output:
        out = args.output
    elif args.output_dir:
        out = args.output_dir / f"{args.resume.stem}.json"
    else:
        out = args.resume.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blanks_to_null(parsed.model_dump()), indent=2, ensure_ascii=False), encoding="utf-8")
    n_roles = sum(len(c.roles) for c in parsed.companies)
    n_bullets = sum(len(r.bullets) for c in parsed.companies for r in c.roles)
    print(
        f"Wrote {out}  ({len(parsed.companies)} companies, {n_roles} roles, {n_bullets} bullets, "
        f"{len(parsed.unclassified)} unclassified)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
