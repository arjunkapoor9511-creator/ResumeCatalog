"""Story bank: SQLite store for parsed resumes.

Usage:
    python story_bank.py import <parsed_json_folder> [--db C:\\Resumedump\\story_bank.db]
    python story_bank.py stats [--db ...]

Data model:
    resumes     one row per parsed resume (raw JSON kept)
    companies   one row per company name (as written; case-insensitive match)
    positions   one row per (company, title)
    bullets     one row per UNIQUE bullet text (deduplicated), tagged with the company,
                position and resume where it was first seen. parent_id (NULL for now)
                links a variant to its root story later.
    bullet_resumes  every resume a bullet appears in (keeps usage history after dedup).

Dedup key: exact bullet text after trimming and collapsing whitespace. Different
wording = a different bullet (variants are what the story bank tracks).

Import is idempotent: a resume whose file name is already in the DB is skipped.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(r"C:\Resumedump\story_bank.db")

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS resumes (
    id          INTEGER PRIMARY KEY,
    file_name   TEXT NOT NULL UNIQUE,
    parsed_json TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS companies (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE
);

CREATE TABLE IF NOT EXISTS positions (
    id         INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id),
    title      TEXT NOT NULL COLLATE NOCASE,
    UNIQUE (company_id, title)
);

CREATE TABLE IF NOT EXISTS bullets (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL UNIQUE,           -- dedup key (whitespace-normalised)
    company_id  INTEGER NOT NULL REFERENCES companies(id),
    position_id INTEGER NOT NULL REFERENCES positions(id),
    resume_id   INTEGER NOT NULL REFERENCES resumes(id),  -- first resume it appeared in
    parent_id   INTEGER REFERENCES bullets(id), -- NULL = root; set later when grouping variants
    source      TEXT NOT NULL DEFAULT 'resume' CHECK (source IN ('resume', 'generated', 'manual')),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bullet_resumes (
    bullet_id   INTEGER NOT NULL REFERENCES bullets(id),
    resume_id   INTEGER NOT NULL REFERENCES resumes(id),
    position_id INTEGER NOT NULL REFERENCES positions(id),  -- position it sat under on that resume
    sort_order  INTEGER NOT NULL,                           -- order within the role on that resume
    PRIMARY KEY (bullet_id, resume_id)
);

CREATE INDEX IF NOT EXISTS ix_bullets_company  ON bullets(company_id);
CREATE INDEX IF NOT EXISTS ix_bullets_position ON bullets(position_id);
CREATE INDEX IF NOT EXISTS ix_bullets_resume   ON bullets(resume_id);
CREATE INDEX IF NOT EXISTS ix_bullets_parent   ON bullets(parent_id);
CREATE INDEX IF NOT EXISTS ix_bullet_resumes_resume ON bullet_resumes(resume_id);
"""


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    return conn


def _get_or_create(conn: sqlite3.Connection, select: str, insert: str, args: tuple) -> int:
    row = conn.execute(select, args).fetchone()
    if row:
        return row[0]
    return conn.execute(insert, args).lastrowid


_LEADING_GLYPHS = "•·▪‣◦●○■□➢►-–—* \t"


def normalise(text: str) -> str:
    """Dedup key: strip leading bullet glyphs (the parser sometimes copies the '•'
    into the text) and collapse whitespace. Wording is otherwise untouched."""
    return " ".join(text.lstrip(_LEADING_GLYPHS).split())


def import_resume(conn: sqlite3.Connection, path: Path) -> tuple[int, int] | None:
    """Import one parsed resume JSON. Returns (new bullets, repeats of existing bullets),
    or None if the resume was already imported."""
    if conn.execute("SELECT 1 FROM resumes WHERE file_name = ?", (path.name,)).fetchone():
        return None
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    added = repeated = 0
    with conn:  # one transaction per resume
        resume_id = conn.execute(
            "INSERT INTO resumes (file_name, parsed_json) VALUES (?, ?)", (path.name, raw)
        ).lastrowid
        for company in data.get("companies", []):
            name = (company.get("name") or "").strip() or "(unknown company)"
            company_id = _get_or_create(
                conn,
                "SELECT id FROM companies WHERE name = ?",
                "INSERT INTO companies (name) VALUES (?)",
                (name,),
            )
            for role in company.get("roles", []):
                title = (role.get("title") or "").strip() or "(unknown position)"
                position_id = _get_or_create(
                    conn,
                    "SELECT id FROM positions WHERE company_id = ? AND title = ?",
                    "INSERT INTO positions (company_id, title) VALUES (?, ?)",
                    (company_id, title),
                )
                for order, bullet in enumerate(role.get("bullets", [])):
                    text = normalise(bullet.get("text") or "")
                    if not text:
                        continue
                    row = conn.execute("SELECT id FROM bullets WHERE text = ?", (text,)).fetchone()
                    if row:
                        bullet_id = row[0]
                        repeated += 1
                    else:
                        bullet_id = conn.execute(
                            "INSERT INTO bullets (text, company_id, position_id, resume_id) VALUES (?, ?, ?, ?)",
                            (text, company_id, position_id, resume_id),
                        ).lastrowid
                        added += 1
                    # A bullet repeated within the same resume is recorded once.
                    conn.execute(
                        "INSERT OR IGNORE INTO bullet_resumes (bullet_id, resume_id, position_id, sort_order) "
                        "VALUES (?, ?, ?, ?)",
                        (bullet_id, resume_id, position_id, order),
                    )
    return added, repeated


def cmd_import(args) -> int:
    folder: Path = args.folder
    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 2
    files = sorted(p for p in folder.glob("*.json") if not p.name.startswith("batch_"))
    conn = connect(args.db)
    imported = skipped = failed = 0
    for path in files:
        try:
            n = import_resume(conn, path)
        except (json.JSONDecodeError, sqlite3.Error, AttributeError, TypeError) as e:
            print(f"FAIL  {path.name}: {e}")
            failed += 1
            continue
        if n is None:
            skipped += 1
        else:
            imported += 1
            print(f"OK    {path.name}  ({n[0]} new bullets, {n[1]} repeats linked)")
    print(f"\nImported {imported}, skipped {skipped} already in DB, {failed} failed  ->  {args.db}")
    print_stats(conn)
    return 1 if failed else 0


def print_stats(conn: sqlite3.Connection) -> None:
    q = lambda sql: conn.execute(sql).fetchone()[0]
    print(
        f"DB totals: {q('SELECT COUNT(*) FROM resumes')} resumes, "
        f"{q('SELECT COUNT(*) FROM companies')} companies, "
        f"{q('SELECT COUNT(*) FROM positions')} positions, "
        f"{q('SELECT COUNT(*) FROM bullets')} unique bullets "
        f"(used {q('SELECT COUNT(*) FROM bullet_resumes')} times across resumes)"
    )


def cmd_stats(args) -> int:
    if not args.db.exists():
        print(f"No database at {args.db}", file=sys.stderr)
        return 1
    print_stats(connect(args.db))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"SQLite file (default {DEFAULT_DB})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="Import a folder of parsed resume JSONs")
    imp.add_argument("folder", type=Path)
    imp.set_defaults(func=cmd_import)
    st = sub.add_parser("stats", help="Show row counts")
    st.set_defaults(func=cmd_stats)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
