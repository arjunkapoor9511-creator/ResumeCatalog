"""Opus-curated story bank: canonicalise companies/positions and group bullet variants.

Usage (add --db PATH before the command to target another database):
    python curate.py entities         # Opus groups company names, then position titles
    python curate.py stories          # Opus groups bullets into stories, picks a root version
    python curate.py add RESUME.json  # incremental: add one parsed resume to the curated bank
    python curate.py status           # counts by kind / confidence / status
    python curate.py review           # resolve pending (medium/low) matches interactively
    python curate.py apply            # write accepted decisions back to the original tables

High-confidence groups are auto-accepted; medium/low land in `matches` as pending.
Nothing in the original tables (companies, positions, bullets, bullet_resumes) changes
until `apply`, which refuses to run while anything is pending and backs up the DB first.

Root rule for a story: the version used in the most resumes; on a tie, the most
flattering version (Opus's pick). The most-used rule is enforced in code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

from story_bank import DEFAULT_DB, connect, normalise

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
Confidence = Literal["high", "medium", "low"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id            INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('company', 'position', 'story')),
    confidence    TEXT NOT NULL CHECK (confidence IN ('high', 'medium', 'low')),
    summary       TEXT NOT NULL,
    rationale     TEXT NOT NULL,
    proposal_json TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('auto_accepted', 'pending', 'accepted', 'rejected')),
    decided_at    TEXT
);
CREATE TABLE IF NOT EXISTS curated_companies (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE
);
CREATE TABLE IF NOT EXISTS company_map (       -- original company id -> curated company
    company_id         INTEGER PRIMARY KEY,    -- no FK: apply deletes merged originals
    curated_company_id INTEGER NOT NULL REFERENCES curated_companies(id),
    match_id           INTEGER REFERENCES matches(id)   -- NULL = unchanged singleton
);
CREATE TABLE IF NOT EXISTS curated_positions (
    id                 INTEGER PRIMARY KEY,
    curated_company_id INTEGER NOT NULL REFERENCES curated_companies(id),
    title              TEXT NOT NULL COLLATE NOCASE,
    UNIQUE (curated_company_id, title)
);
CREATE TABLE IF NOT EXISTS position_map (
    position_id         INTEGER PRIMARY KEY,
    curated_position_id INTEGER NOT NULL REFERENCES curated_positions(id),
    match_id            INTEGER REFERENCES matches(id)
);
CREATE TABLE IF NOT EXISTS stories (
    id                 INTEGER PRIMARY KEY,
    group_label        TEXT NOT NULL,          -- company group the bullets were sent under
    root_bullet_id     INTEGER NOT NULL REFERENCES bullets(id),
    shared_facts       TEXT NOT NULL,
    root_reason        TEXT NOT NULL,
    match_id           INTEGER REFERENCES matches(id)   -- NULL = single-bullet story
);
CREATE TABLE IF NOT EXISTS story_members (
    bullet_id INTEGER PRIMARY KEY REFERENCES bullets(id),
    story_id  INTEGER NOT NULL REFERENCES stories(id)
);
CREATE TABLE IF NOT EXISTS curation_runs (
    id            INTEGER PRIMARY KEY,
    step          TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

ACCEPTED = ("auto_accepted", "accepted")

# --------------------------------------------------------------------------
# Structured output models (no nullable fields: unions bloat the grammar)
# --------------------------------------------------------------------------


class CompanyGroup(BaseModel):
    canonical_name: str = Field(description="Employer's common short name in normal casing, no tags, e.g. 'OutSystems'.")
    member_company_ids: list[int]
    confidence: Confidence
    rationale: str


class CompanyGrouping(BaseModel):
    groups: list[CompanyGroup]


class PositionGroup(BaseModel):
    canonical_title: str
    member_position_ids: list[int]
    confidence: Confidence
    rationale: str


class PositionGrouping(BaseModel):
    groups: list[PositionGroup]


class StoryGroup(BaseModel):
    member_ids: list[int]
    root_id: int
    root_reason: str = Field(description="Why this version is the root (usage, or why it is the most flattering on a tie).")
    shared_facts: str = Field(description="Distinctive facts shared by the members (client, metrics, scope). Empty for single-bullet groups.")
    confidence: Confidence


class StoryGrouping(BaseModel):
    groups: list[StoryGroup]


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

COMPANY_PROMPT = """You canonicalise employer names taken from one person's resumes.

Group names that refer to the same employer: short vs full/legal names (e.g. "X" and
"X Software"), names with a practice or division tag (e.g. "Firm (Data & Analytics practice)").
Never merge different employers. A label that isn't a real employer name (e.g. "Various
Contracts") may only be merged if its titles and bullets clearly show the same employer;
otherwise keep it on its own.

canonical_name: the employer's common short name in normal casing, without tags.
confidence: high = certainly the same employer (a name kept on its own is high);
medium = probably; low = unsure.
Every company id must appear in exactly one group."""

POSITION_PROMPT = """You canonicalise job titles one person held at a single employer.

Group titles that denote the same role and differ only in tailoring or formatting: level
codes like "(L61)", team or segment suffixes like ", SME&C" or " - Strategic Media Clients",
"AI"/"Cloud & AI" prefix variations, punctuation or casing. Keep genuinely different roles
separate: different seniority (Analyst vs Consultant vs Senior Consultant), different jobs
(Consultant vs Solutions Architect vs GTM Manager), intern vs full-time, different practice
when it is a different job.

canonical_title: the clearest common form of the title, no level codes.
confidence: high = certainly the same role (a title kept on its own is high);
medium = probably; low = unsure.
Every position id must appear in exactly one group."""

STORY_PROMPT = """You organise one person's resume bullets from a single employer into stories.

Group bullets that describe the same underlying accomplishment: the same project, client,
deal or initiative and its outcome, even when rewritten with a different emphasis (technical
vs people/leadership vs business impact), a different selection of metrics, or a different
length. Do NOT group bullets that only share a theme (two different chatbot projects, two
different deals). Shared distinctive facts (client, metric values, team size, deal size) are
strong evidence. If metrics conflict for what is otherwise clearly the same story, still group
them with confidence medium and name the conflict in shared_facts.

root_id: the member with the highest times_used. If several members tie for the highest
times_used, choose the most flattering of those: strongest quantified outcome, largest scope,
clearest ownership.
shared_facts: the facts that make the members one story; empty string for single-bullet groups.
confidence: high = clearly the same story (a single-bullet group is high); medium = probably;
low = possibly.
Every bullet id must appear in exactly one group."""


# --------------------------------------------------------------------------
# Opus call
# --------------------------------------------------------------------------


def _consume_live(stream, step: str) -> None:
    """Print streaming progress: thinking, then the answer as it arrives."""
    phase, chars, shown = "", 0, 0
    for event in stream:
        if event.type == "content_block_start" and getattr(event.content_block, "type", "") == "thinking" and phase != "thinking":
            phase = "thinking"
            print(f"  {step}: Opus is thinking...", flush=True)
        elif event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
            if phase != "answer":
                phase = "answer"
                print(f"  {step}: writing answer", end="", flush=True)
            chars += len(event.delta.text)
            if chars - shown >= 500:          # a dot per ~500 chars keeps logs readable
                shown = chars
                print(".", end="", flush=True)
    if phase == "answer":
        print(f" {chars} chars", flush=True)


def call_opus(conn, client, step: str, system: str, payload: str, model_cls, validate, live: bool = False) -> BaseModel:
    """Structured-output call; `validate(parsed)` returns an error string or ''. One retry
    with the error fed back if the output breaks the id rules. `live` prints streaming progress."""
    messages = [{"role": "user", "content": payload}]
    for attempt in (1, 2):
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=64000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=system,
            messages=messages,
            output_config={
                "effort": "high",
                "format": {"type": "json_schema", "schema": anthropic.transform_schema(model_cls)},
            },
        ) as stream:
            if live:
                _consume_live(stream, step)
            msg = stream.get_final_message()
        conn.execute(
            "INSERT INTO curation_runs (step, model, input_tokens, output_tokens) VALUES (?, ?, ?, ?)",
            (step, msg.model, msg.usage.input_tokens, msg.usage.output_tokens),
        )
        conn.commit()
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"{step}: model declined ({getattr(msg.stop_details, 'category', None)})")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError(f"{step}: output hit max_tokens")
        # After a fallback switch, the answer is the text after the last fallback block.
        blocks = list(msg.content)
        last_fb = max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=-1)
        text = "".join(b.text for b in blocks[last_fb + 1:] if b.type == "text")
        try:
            parsed = model_cls.model_validate_json(text)
        except ValidationError as e:
            raise RuntimeError(f"{step}: output failed schema validation: {e}") from e
        error = validate(parsed)
        if not error:
            return parsed
        print(f"  {step}: attempt {attempt} broke the rules ({error}); retrying" if attempt == 1 else "")
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"That grouping is invalid: {error}. Return the corrected full grouping."},
        ]
    raise RuntimeError(f"{step}: invalid grouping after retry: {error}")


def check_partition(groups_ids: list[list[int]], expected: set[int]) -> str:
    seen = Counter(i for ids in groups_ids for i in ids)
    dup = sorted(i for i, n in seen.items() if n > 1)
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    problems = []
    if dup: problems.append(f"ids in more than one group: {dup}")
    if missing: problems.append(f"ids missing: {missing}")
    if extra: problems.append(f"unknown ids: {extra}")
    if any(not ids for ids in groups_ids): problems.append("empty group")
    return "; ".join(problems)


def add_match(conn, kind, confidence, summary, rationale, proposal) -> tuple[int, str]:
    status = "auto_accepted" if confidence == "high" else "pending"
    mid = conn.execute(
        "INSERT INTO matches (kind, confidence, summary, rationale, proposal_json, status) VALUES (?, ?, ?, ?, ?, ?)",
        (kind, confidence, summary, rationale, json.dumps(proposal, ensure_ascii=False), status),
    ).lastrowid
    return mid, status


def setup(db: Path) -> sqlite3.Connection:
    conn = connect(db)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA foreign_keys = ON")
    # Migration: per-member decision for bullets added incrementally to an existing story.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(story_members)")}
    if "match_id" not in cols:
        conn.execute("ALTER TABLE story_members ADD COLUMN match_id INTEGER REFERENCES matches(id)")
    # SQLite reuses deleted row ids, so map rows for companies/positions that `apply` merged
    # away must not linger (a new company could inherit a stale mapping).
    conn.execute("DELETE FROM company_map WHERE company_id NOT IN (SELECT id FROM companies)")
    conn.execute("DELETE FROM position_map WHERE position_id NOT IN (SELECT id FROM positions)")
    conn.commit()
    return conn


def _accepted(conn, mid) -> bool:
    return mid is None or conn.execute("SELECT status FROM matches WHERE id = ?", (mid,)).fetchone()[0] in ACCEPTED


def _usage(conn, bullet_id: int) -> int:
    return conn.execute("SELECT COUNT(*) FROM bullet_resumes WHERE bullet_id = ?", (bullet_id,)).fetchone()[0]


def sync_story(conn, sid: int) -> int:
    """Make bullets.parent_id reflect story `sid` given current decisions. Members whose own
    (incremental) match was rejected are split off into single-bullet stories. Returns the
    number of variants linked."""
    root, mid, label = conn.execute("SELECT root_bullet_id, match_id, group_label FROM stories WHERE id = ?", (sid,)).fetchone()
    rows = conn.execute("SELECT bullet_id, match_id FROM story_members WHERE story_id = ?", (sid,)).fetchall()
    for bid, mmid in rows:
        if mmid is not None and conn.execute("SELECT status FROM matches WHERE id = ?", (mmid,)).fetchone()[0] == "rejected":
            new_sid = conn.execute(
                "INSERT INTO stories (group_label, root_bullet_id, shared_facts, root_reason, match_id) VALUES (?, ?, '', 'only version', NULL)",
                (label, bid)).lastrowid
            conn.execute("UPDATE story_members SET story_id = ?, match_id = NULL WHERE bullet_id = ?", (new_sid, bid))
            conn.execute("UPDATE bullets SET parent_id = NULL WHERE id = ?", (bid,))
    members = [b for b, m in conn.execute("SELECT bullet_id, match_id FROM story_members WHERE story_id = ?", (sid,))
               if _accepted(conn, m)]
    if not members:
        return 0
    if not _accepted(conn, mid):          # whole story rejected / pending: members stand alone
        conn.execute(f"UPDATE bullets SET parent_id = NULL WHERE id IN ({','.join('?' * len(members))})", members)
        return 0
    if root not in members:               # root itself was split off
        root = max(members, key=lambda b: (_usage(conn, b), -b))
        conn.execute("UPDATE stories SET root_bullet_id = ? WHERE id = ?", (root, sid))
    conn.execute("UPDATE bullets SET parent_id = NULL WHERE id = ?", (root,))
    linked = 0
    for b in members:
        if b != root:
            conn.execute("UPDATE bullets SET parent_id = ? WHERE id = ?", (root, b))
            linked += 1
    return linked


def promote_most_used(conn, sid: int) -> bool:
    """If a member is now used strictly more than the root (after new resume links), make it root."""
    root = conn.execute("SELECT root_bullet_id FROM stories WHERE id = ?", (sid,)).fetchone()[0]
    members = [b for (b,) in conn.execute("SELECT bullet_id FROM story_members WHERE story_id = ?", (sid,))]
    best = max(members, key=lambda b: (_usage(conn, b), b == root))
    if best != root and _usage(conn, best) > _usage(conn, root):
        conn.execute("UPDATE stories SET root_bullet_id = ?, root_reason = ? WHERE id = ?",
                     (best, f"most-used version ({_usage(conn, best)} resumes); promoted after new resume", sid))
        return True
    return False


# --------------------------------------------------------------------------
# entities: companies, then positions
# --------------------------------------------------------------------------


def cmd_entities(conn, client, args) -> int:
    if conn.execute("SELECT 1 FROM company_map").fetchone():
        if not args.redo:
            print("Entities already curated. Use `entities --redo` to discard and redo (also clears stories).")
            return 1
        clear(conn, "entities")

    companies = conn.execute("SELECT id, name FROM companies ORDER BY id").fetchall()
    lines = []
    for cid, name in companies:
        titles = [t for (t,) in conn.execute("SELECT title FROM positions WHERE company_id = ?", (cid,))]
        samples = [t[:160] for (t,) in conn.execute("SELECT text FROM bullets WHERE company_id = ? LIMIT 2", (cid,))]
        lines.append(f"id={cid} | name={name!r} | titles={titles} | sample bullets={samples}")
    ids = {cid for cid, _ in companies}
    names = dict(companies)

    def validate_companies(g: CompanyGrouping) -> str:
        err = check_partition([x.member_company_ids for x in g.groups], ids)
        dup_names = [n for n, c in Counter(x.canonical_name.lower() for x in g.groups).items() if c > 1]
        return "; ".join(filter(None, [err, f"duplicate canonical names {dup_names}" if dup_names else ""]))

    print(f"Companies: sending {len(companies)} names to {MODEL} ...")
    grouping = call_opus(conn, client, "companies", COMPANY_PROMPT, "\n".join(lines), CompanyGrouping, validate_companies)

    with conn:
        for g in grouping.groups:
            ccid = conn.execute("INSERT INTO curated_companies (name) VALUES (?)", (g.canonical_name,)).lastrowid
            members = [names[i] for i in g.member_company_ids]
            changed = len(members) > 1 or members[0] != g.canonical_name
            mid = None
            if changed:
                mid, status = add_match(conn, "company", g.confidence, f"{' + '.join(members)} -> {g.canonical_name}",
                                        g.rationale, g.model_dump())
                print(f"  [{status:13}] {g.confidence:6} {' + '.join(members)} -> {g.canonical_name}")
            for cid in g.member_company_ids:
                conn.execute("INSERT INTO company_map VALUES (?, ?, ?)", (cid, ccid, mid))

    # Positions: one call per proposed curated company.
    for ccid, cname in conn.execute("SELECT id, name FROM curated_companies ORDER BY id").fetchall():
        positions = conn.execute(
            """SELECT p.id, p.title, (SELECT COUNT(*) FROM bullet_resumes br WHERE br.position_id = p.id)
               FROM positions p JOIN company_map m ON m.company_id = p.company_id
               WHERE m.curated_company_id = ? ORDER BY p.id""", (ccid,)).fetchall()
        if not positions:
            continue
        titles = {pid: t for pid, t, _ in positions}
        if len(positions) == 1:
            g = PositionGroup(canonical_title=positions[0][1], member_position_ids=[positions[0][0]],
                              confidence="high", rationale="only title")
            groups = [g]
        else:
            print(f"Positions at {cname}: sending {len(positions)} titles ...")
            payload = f"Employer: {cname}\n" + "\n".join(f"id={pid} | title={t!r} | bullets on resumes={n}" for pid, t, n in positions)
            groups = call_opus(conn, client, f"positions:{cname}", POSITION_PROMPT, payload, PositionGrouping,
                               lambda g: check_partition([x.member_position_ids for x in g.groups], set(titles))).groups
        with conn:
            for g in groups:
                cpid = conn.execute(
                    "INSERT OR IGNORE INTO curated_positions (curated_company_id, title) VALUES (?, ?)", (ccid, g.canonical_title)
                ).lastrowid or conn.execute(
                    "SELECT id FROM curated_positions WHERE curated_company_id = ? AND title = ?", (ccid, g.canonical_title)
                ).fetchone()[0]
                members = [titles[i] for i in g.member_position_ids]
                changed = len(members) > 1 or members[0] != g.canonical_title
                mid = None
                if changed:
                    mid, status = add_match(conn, "position", g.confidence,
                                            f"{cname}: {' + '.join(members)} -> {g.canonical_title}", g.rationale, g.model_dump())
                    print(f"  [{status:13}] {g.confidence:6} {' + '.join(members)} -> {g.canonical_title}")
                for pid in g.member_position_ids:
                    conn.execute("INSERT INTO position_map VALUES (?, ?, ?)", (pid, cpid, mid))
    print_status(conn)
    return 0


# --------------------------------------------------------------------------
# stories
# --------------------------------------------------------------------------


def story_groups(conn) -> dict[str, list[int]]:
    """Company groups to send bullets under: accepted curated companies; companies whose
    merge is pending/rejected are sent under their original name."""
    out: dict[str, list[int]] = {}
    for cid, cname, ccname, status in conn.execute(
        """SELECT c.id, c.name, cc.name, m.status FROM companies c
           JOIN company_map cm ON cm.company_id = c.id
           JOIN curated_companies cc ON cc.id = cm.curated_company_id
           LEFT JOIN matches m ON m.id = cm.match_id"""):
        label = ccname if status is None or status in ACCEPTED else cname
        out.setdefault(label, []).append(cid)
    return out


def cmd_stories(conn, client, args) -> int:
    if not conn.execute("SELECT 1 FROM company_map").fetchone():
        print("Run `entities` first.")
        return 1
    if conn.execute("SELECT 1 FROM stories").fetchone():
        if not args.redo:
            print("Stories already curated. Use `stories --redo` to discard and redo.")
            return 1
        clear(conn, "stories")

    for label, company_ids in story_groups(conn).items():
        marks = ",".join("?" * len(company_ids))
        bullets = conn.execute(
            f"""SELECT b.id, p.title, (SELECT COUNT(*) FROM bullet_resumes br WHERE br.bullet_id = b.id), b.text
                FROM bullets b JOIN positions p ON p.id = b.position_id
                WHERE b.company_id IN ({marks}) ORDER BY b.id""", company_ids).fetchall()
        if not bullets:
            continue
        used = {bid: n for bid, _, n, _ in bullets}
        if len(bullets) == 1:
            groups = [StoryGroup(member_ids=[bullets[0][0]], root_id=bullets[0][0], root_reason="only version",
                                 shared_facts="", confidence="high")]
        else:
            print(f"Stories at {label}: sending {len(bullets)} bullets ...")
            payload = f"Employer: {label}\n" + "\n".join(
                f"id={bid} | position={t!r} | times_used={n} | {text}" for bid, t, n, text in bullets)

            def validate(g: StoryGrouping) -> str:
                err = check_partition([x.member_ids for x in g.groups], set(used))
                bad_root = [x.root_id for x in g.groups if x.root_id not in x.member_ids]
                return "; ".join(filter(None, [err, f"root_id not in its group: {bad_root}" if bad_root else ""]))

            groups = call_opus(conn, client, f"stories:{label}", STORY_PROMPT, payload, StoryGrouping, validate).groups

        with conn:
            multi = 0
            for g in groups:
                # Enforce the most-used rule in code; Opus's pick stands only on a tie.
                top = max(used[i] for i in g.member_ids)
                tied = [i for i in g.member_ids if used[i] == top]
                root, reason = g.root_id, g.root_reason
                if root not in tied:
                    root = min(tied)
                    reason = f"most-used version ({top} resumes); set by rule, overriding model pick {g.root_id}"
                mid = None
                if len(g.member_ids) > 1:
                    multi += 1
                    proposal = g.model_dump() | {"root_id": root, "root_reason": reason}
                    mid, _ = add_match(conn, "story", g.confidence,
                                       f"{label}: {len(g.member_ids)} versions, root #{root}", g.shared_facts, proposal)
                sid = conn.execute(
                    "INSERT INTO stories (group_label, root_bullet_id, shared_facts, root_reason, match_id) VALUES (?, ?, ?, ?, ?)",
                    (label, root, g.shared_facts, reason, mid)).lastrowid
                for bid in g.member_ids:
                    conn.execute("INSERT INTO story_members VALUES (?, ?)", (bid, sid))
            print(f"  {len(bullets)} bullets -> {len(groups)} stories ({multi} with variants)")
    print_status(conn)
    return 0


# --------------------------------------------------------------------------
# add: incremental import of one parsed resume
# --------------------------------------------------------------------------


class PositionResolution(BaseModel):
    key: str
    existing_position_id: int = Field(description="Matching existing position id, or 0 if it is a new role.")
    canonical_title: str
    confidence: Confidence
    rationale: str


class CompanyResolution(BaseModel):
    key: str
    existing_company_id: int = Field(description="Matching existing company id, or 0 if it is a new employer.")
    canonical_name: str
    confidence: Confidence
    rationale: str
    positions: list[PositionResolution]


class EntityResolution(BaseModel):
    companies: list[CompanyResolution]


class Assignment(BaseModel):
    bullet_id: int
    existing_story_id: int = Field(description="Story the bullet is a version of, or 0 if none.")
    new_group: str = Field(description="When existing_story_id is 0: a short label; new bullets sharing a label form one new story.")
    new_group_root: bool = Field(description="True for the most flattering bullet of a new group.")
    shared_facts: str
    confidence: Confidence


class AssignmentResult(BaseModel):
    assignments: list[Assignment]


RESOLVE_PROMPT = """You map the employers and job titles from one newly parsed resume onto an
existing, already-curated list for the same person.

For each new company (key c#): existing_company_id = the existing employer it is (same rules as
before: short vs full names and practice/division tags are the same employer; never merge
different employers), or 0 if it is a new employer. canonical_name = the existing name when
matched, else the employer's common short name without tags.
For each of its titles (key c#.t#): existing_position_id = an existing title AT THAT SAME
EXISTING COMPANY denoting the same role (ignore level codes, team/segment suffixes, AI/Cloud
prefix tailoring; keep different seniority or different jobs apart), or 0 if new.
canonical_title = the existing title when matched, else the clearest form of the new title.
If the company is new (0), every one of its titles is 0.
confidence: high = certain, medium = probably, low = unsure. Cover every key exactly once."""

ASSIGN_PROMPT = """You add newly imported resume bullets to one person's curated story bank for a
single employer.

A story is one underlying accomplishment (same project, client, deal or initiative and outcome).
Each existing story is shown with its root version and its variants. For each NEW bullet:
- existing_story_id = the story it is another version of, even if rewritten with a different
  emphasis (technical vs people/leadership vs business impact), different metrics or length;
  0 if it is not a version of any existing story. Shared theme alone is not enough.
- If 0: new_group = a short label. New bullets that are versions of each other share a label;
  otherwise each gets its own. new_group_root = true for the most flattering bullet of each new
  group (strongest quantified outcome, scope, ownership), false otherwise.
- shared_facts = the facts linking it to the story or group; empty string if none.
- confidence: high = clearly the same story (or clearly a brand-new story), medium = probably,
  low = possibly. If metrics conflict with the story, use medium and name the conflict.
Every new bullet id must appear exactly once."""


def _curated_company_of(conn, company_id: int, fallback_name: str) -> int:
    row = conn.execute("SELECT curated_company_id FROM company_map WHERE company_id = ?", (company_id,)).fetchone()
    if row:
        return row[0]
    conn.execute("INSERT OR IGNORE INTO curated_companies (name) VALUES (?)", (fallback_name,))
    ccid = conn.execute("SELECT id FROM curated_companies WHERE name = ?", (fallback_name,)).fetchone()[0]
    conn.execute("INSERT INTO company_map VALUES (?, ?, NULL)", (company_id, ccid))
    return ccid


def _curated_position_of(conn, position_id: int, ccid: int, fallback_title: str) -> int:
    row = conn.execute("SELECT curated_position_id FROM position_map WHERE position_id = ?", (position_id,)).fetchone()
    if row:
        return row[0]
    conn.execute("INSERT OR IGNORE INTO curated_positions (curated_company_id, title) VALUES (?, ?)", (ccid, fallback_title))
    cpid = conn.execute("SELECT id FROM curated_positions WHERE curated_company_id = ? AND title = ?",
                        (ccid, fallback_title)).fetchone()[0]
    conn.execute("INSERT INTO position_map VALUES (?, ?, NULL)", (position_id, cpid))
    return cpid


def _new_company(conn, name: str) -> int:
    row = conn.execute("SELECT id FROM companies WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    cid = conn.execute("INSERT INTO companies (name) VALUES (?)", (name,)).lastrowid
    conn.execute("DELETE FROM company_map WHERE company_id = ?", (cid,))   # stale row from a reused id
    return cid


def _new_position(conn, company_id: int, title: str) -> int:
    row = conn.execute("SELECT id FROM positions WHERE company_id = ? AND title = ?", (company_id, title)).fetchone()
    if row:
        return row[0]
    pid = conn.execute("INSERT INTO positions (company_id, title) VALUES (?, ?)", (company_id, title)).lastrowid
    conn.execute("DELETE FROM position_map WHERE position_id = ?", (pid,))
    return pid


def resolve_entities(conn, client, data: dict) -> dict[tuple[int, int], tuple[int, int]]:
    """Map (company index, role index) in the resume JSON to (company_id, position_id), creating
    rows / pending matches as needed. Runs inside the caller's transaction for writes."""
    existing = {cid: name for cid, name in conn.execute("SELECT id, name FROM companies")}
    lower_company = {name.lower(): cid for cid, name in existing.items()}
    titles_of = {cid: {t.lower(): pid for pid, t in conn.execute("SELECT id, title FROM positions WHERE company_id = ?", (cid,))}
                 for cid in existing}
    result: dict[tuple[int, int], tuple[int, int]] = {}
    unresolved = []  # (ci, name, [(ri, title)])
    for ci, comp in enumerate(data.get("companies", [])):
        name = (comp.get("name") or "").strip() or "(unknown company)"
        roles = [(ri, (r.get("title") or "").strip() or "(unknown position)") for ri, r in enumerate(comp.get("roles", []))]
        cid = lower_company.get(name.lower())
        if cid and all(t.lower() in titles_of[cid] for _, t in roles):
            for ri, t in roles:
                result[(ci, ri)] = (cid, titles_of[cid][t.lower()])
        else:
            unresolved.append((ci, name, roles))
    if not unresolved:
        print("Companies/positions: all match existing names exactly (no Opus call).")
        return result

    payload = ["EXISTING:"] + [
        f"company id={cid} | {name!r} | titles: " + "; ".join(f"id={pid} {t!r}" for pid, t in
                                                             conn.execute("SELECT id, title FROM positions WHERE company_id = ?", (cid,)))
        for cid, name in existing.items()]
    payload.append("\nNEW (from the resume being added):")
    keys: set[str] = set()
    for ci, name, roles in unresolved:
        keys.add(f"c{ci}")
        payload.append(f"key=c{ci} | company {name!r}")
        for ri, t in roles:
            keys.add(f"c{ci}.t{ri}")
            payload.append(f"   key=c{ci}.t{ri} | title {t!r}")
    pos_company = {pid: cid for cid, pids in titles_of.items() for pid in pids.values()}

    def validate(r: EntityResolution) -> str:
        got = [c.key for c in r.companies] + [p.key for c in r.companies for p in c.positions]
        problems = []
        if sorted(got) != sorted(keys):
            problems.append(f"keys must be exactly {sorted(keys)}, got {sorted(got)}")
        for c in r.companies:
            if c.existing_company_id and c.existing_company_id not in existing:
                problems.append(f"unknown company id {c.existing_company_id}")
            for p in c.positions:
                if p.existing_position_id and pos_company.get(p.existing_position_id) != c.existing_company_id:
                    problems.append(f"position {p.existing_position_id} is not at company {c.existing_company_id}")
        return "; ".join(problems)

    print(f"Companies/positions: resolving {len(unresolved)} against the curated list ...")
    res = call_opus(conn, client, "add:entities", RESOLVE_PROMPT, "\n".join(payload), EntityResolution, validate, live=True)
    by_key = {c.key: c for c in res.companies}
    for ci, name, roles in unresolved:
        c = by_key[f"c{ci}"]
        if c.existing_company_id and c.confidence == "high":
            cid, direct = c.existing_company_id, True
        elif c.existing_company_id:                 # probable match: add as written, merge pending review
            cid, direct = _new_company(conn, name), False
            if cid != c.existing_company_id and not conn.execute("SELECT 1 FROM company_map WHERE company_id = ?", (cid,)).fetchone():
                mid, status = add_match(conn, "company", c.confidence, f"{name} -> {existing[c.existing_company_id]}",
                                        c.rationale, c.model_dump())
                conn.execute("INSERT INTO company_map VALUES (?, ?, ?)",
                             (cid, _curated_company_of(conn, c.existing_company_id, existing[c.existing_company_id]), mid))
                print(f"  [{status:13}] {c.confidence:6} company {name} -> {existing[c.existing_company_id]}")
        else:                                       # new employer
            cid, direct = _new_company(conn, c.canonical_name if c.confidence == "high" else name), False
            _curated_company_of(conn, cid, c.canonical_name)
            print(f"  [new          ] company {name}")
        direct = direct or cid == c.existing_company_id   # same row as the existing company
        if direct and c.confidence == "high":
            print(f"  [auto_accepted] high   company {name} -> {existing[cid]}")
        ccid = _curated_company_of(conn, cid, c.canonical_name)
        pos = {p.key: p for p in c.positions}
        for ri, t in roles:
            p = pos[f"c{ci}.t{ri}"]
            if p.existing_position_id and p.confidence == "high" and direct:
                pid = p.existing_position_id
                print(f"  [auto_accepted] high   title {t!r} -> {p.canonical_title!r}")
            elif p.existing_position_id:
                pid = _new_position(conn, cid, t)
                if pid != p.existing_position_id and not conn.execute("SELECT 1 FROM position_map WHERE position_id = ?", (pid,)).fetchone():
                    target = conn.execute("SELECT title, company_id FROM positions WHERE id = ?", (p.existing_position_id,)).fetchone()
                    mid, status = add_match(conn, "position", p.confidence, f"{name}: {t} -> {target[0]}", p.rationale, p.model_dump())
                    conn.execute("INSERT INTO position_map VALUES (?, ?, ?)",
                                 (pid, _curated_position_of(conn, p.existing_position_id,
                                                            _curated_company_of(conn, target[1], existing[target[1]]), target[0]), mid))
                    print(f"  [{status:13}] {p.confidence:6} title {t!r} -> {target[0]!r}")
            else:
                pid = _new_position(conn, cid, p.canonical_title if p.confidence == "high" else t)
                _curated_position_of(conn, pid, ccid, p.canonical_title)
                print(f"  [new          ] title {t!r}")
            result[(ci, ri)] = (cid, pid)
    return result


def assign_orphans(conn, client) -> None:
    """Put every bullet that is in no story yet into an existing or new story (Opus)."""
    orphans = conn.execute(
        """SELECT b.id, b.company_id, p.title, b.text FROM bullets b JOIN positions p ON p.id = b.position_id
           WHERE b.id NOT IN (SELECT bullet_id FROM story_members) ORDER BY b.id""").fetchall()
    if not orphans:
        print("Stories: no new bullets to place.")
        return
    groups: dict[int, list] = {}
    for row in orphans:
        ccid = _curated_company_of(conn, row[1], conn.execute("SELECT name FROM companies WHERE id = ?", (row[1],)).fetchone()[0])
        groups.setdefault(ccid, []).append(row)
    conn.commit()

    for ccid, rows in groups.items():
        label = conn.execute("SELECT name FROM curated_companies WHERE id = ?", (ccid,)).fetchone()[0]
        stories = conn.execute(
            """SELECT s.id, s.root_bullet_id FROM stories s JOIN bullets r ON r.id = s.root_bullet_id
               JOIN company_map cm ON cm.company_id = r.company_id WHERE cm.curated_company_id = ?""", (ccid,)).fetchall()
        new_ids = {r[0] for r in rows}
        if not stories and len(rows) == 1:
            result = [Assignment(bullet_id=rows[0][0], existing_story_id=0, new_group="only", new_group_root=True,
                                 shared_facts="", confidence="high")]
        else:
            lines = ["EXISTING STORIES:"]
            for sid, root in stories:
                lines.append(f"story_id={sid} | root: {conn.execute('SELECT text FROM bullets WHERE id = ?', (root,)).fetchone()[0]}")
                for (v,) in conn.execute("SELECT b.text FROM story_members sm JOIN bullets b ON b.id = sm.bullet_id "
                                         "WHERE sm.story_id = ? AND b.id != ?", (sid, root)):
                    lines.append(f"      variant: {v}")
            lines.append("\nNEW BULLETS:")
            lines += [f"id={bid} | position={t!r} | {text}" for bid, _, t, text in rows]
            story_ids = {sid for sid, _ in stories}

            def validate(r: AssignmentResult) -> str:
                problems = [check_partition([[a.bullet_id] for a in r.assignments], new_ids)]
                problems += [f"unknown story {a.existing_story_id}" for a in r.assignments
                             if a.existing_story_id and a.existing_story_id not in story_ids]
                problems += [f"bullet {a.bullet_id} needs new_group" for a in r.assignments
                             if not a.existing_story_id and not a.new_group.strip()]
                return "; ".join(p for p in problems if p)

            print(f"Stories at {label}: placing {len(rows)} new bullets among {len(stories)} stories ...")
            result = call_opus(conn, client, f"add:stories:{label}", ASSIGN_PROMPT, "\n".join(lines),
                               AssignmentResult, validate, live=True).assignments

        with conn:
            fresh: dict[str, list[Assignment]] = {}
            for a in result:
                if not a.existing_story_id:
                    fresh.setdefault(a.new_group.strip().lower(), []).append(a)
                    continue
                sid = a.existing_story_id
                root = conn.execute("SELECT root_bullet_id FROM stories WHERE id = ?", (sid,)).fetchone()[0]
                members = [b for (b,) in conn.execute("SELECT bullet_id FROM story_members WHERE story_id = ?", (sid,))]
                mid, status = add_match(conn, "story", a.confidence, f"{label}: add #{a.bullet_id} to story #{sid}", a.shared_facts,
                                        {"member_ids": members + [a.bullet_id], "root_id": root, "added_bullet_id": a.bullet_id,
                                         "story_id": sid, "root_reason": "existing root kept", "confidence": a.confidence})
                conn.execute("INSERT INTO story_members (bullet_id, story_id, match_id) VALUES (?, ?, ?)", (a.bullet_id, sid, mid))
                promote_most_used(conn, sid)
                sync_story(conn, sid)
                print(f"  [{status:13}] {a.confidence:6} #{a.bullet_id} -> story #{sid}")
            for grp in fresh.values():
                ids = [a.bullet_id for a in grp]
                top = max(_usage(conn, b) for b in ids)
                tied = [a for a in grp if _usage(conn, a.bullet_id) == top]
                pick = next((a for a in tied if a.new_group_root), tied[0])
                conf = min((a.confidence for a in grp), key=["low", "medium", "high"].index)
                facts = "; ".join(sorted({a.shared_facts for a in grp if a.shared_facts}))
                mid = None
                if len(ids) > 1:
                    mid, status = add_match(conn, "story", conf, f"{label}: new story, {len(ids)} versions, root #{pick.bullet_id}",
                                            facts, {"member_ids": ids, "root_id": pick.bullet_id,
                                                    "root_reason": "most flattering new version", "confidence": conf})
                sid = conn.execute(
                    "INSERT INTO stories (group_label, root_bullet_id, shared_facts, root_reason, match_id) VALUES (?, ?, ?, ?, ?)",
                    (label, pick.bullet_id, facts, "only version" if len(ids) == 1 else "most flattering new version", mid)).lastrowid
                for b in ids:
                    conn.execute("INSERT INTO story_members (bullet_id, story_id, match_id) VALUES (?, ?, NULL)", (b, sid))
                sync_story(conn, sid)
                print(f"  [new story    ] {len(ids)} version(s): #{', #'.join(map(str, ids))}")


def cmd_add(conn, client, args) -> int:
    if not conn.execute("SELECT 1 FROM stories").fetchone():
        print("No curated stories yet. Run `entities` and `stories` (or `story_bank.py import`) first.")
        return 1
    path: Path = args.resume_json
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 2
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if conn.execute("SELECT 1 FROM resumes WHERE file_name = ?", (path.name,)).fetchone():
        print(f"{path.name} is already imported; placing any unplaced bullets only.")
    else:
        mapping = resolve_entities(conn, client, data)
        new = repeats = 0
        touched: set[int] = set()
        with conn:
            resume_id = conn.execute("INSERT INTO resumes (file_name, parsed_json) VALUES (?, ?)", (path.name, raw)).lastrowid
            for ci, comp in enumerate(data.get("companies", [])):
                for ri, role in enumerate(comp.get("roles", [])):
                    cid, pid = mapping[(ci, ri)]
                    for order, b in enumerate(role.get("bullets", [])):
                        text = normalise(b.get("text") or "")
                        if not text:
                            continue
                        row = conn.execute("SELECT id FROM bullets WHERE text = ?", (text,)).fetchone()
                        if row:
                            bid = row[0]
                            repeats += 1
                            s = conn.execute("SELECT story_id FROM story_members WHERE bullet_id = ?", (bid,)).fetchone()
                            if s:
                                touched.add(s[0])
                        else:
                            bid = conn.execute("INSERT INTO bullets (text, company_id, position_id, resume_id) VALUES (?, ?, ?, ?)",
                                               (text, cid, pid, resume_id)).lastrowid
                            new += 1
                        conn.execute("INSERT OR IGNORE INTO bullet_resumes (bullet_id, resume_id, position_id, sort_order) "
                                     "VALUES (?, ?, ?, ?)", (bid, resume_id, pid, order))
            promoted = sum(promote_most_used(conn, sid) for sid in touched)
            for sid in touched:
                sync_story(conn, sid)
        print(f"Imported {path.name}: {new} new bullets, {repeats} exact repeats linked"
              + (f", {promoted} story root(s) promoted to the now most-used version" if promoted else ""))

    assign_orphans(conn, client)
    print_status(conn)
    return 0


def clear(conn, step: str) -> None:
    with conn:
        conn.execute("DELETE FROM story_members")
        conn.execute("DELETE FROM stories")
        conn.execute("DELETE FROM matches WHERE kind = 'story'")
        if step == "entities":
            for t in ("position_map", "curated_positions", "company_map", "curated_companies"):
                conn.execute(f"DELETE FROM {t}")
            conn.execute("DELETE FROM matches")


# --------------------------------------------------------------------------
# status / review
# --------------------------------------------------------------------------


def print_status(conn) -> None:
    rows = conn.execute("SELECT kind, confidence, status, COUNT(*) FROM matches GROUP BY 1, 2, 3 ORDER BY 1, 2, 3").fetchall()
    print("\nmatches:")
    for kind, conf, status, n in rows:
        print(f"  {kind:8} {conf:6} {status:13} {n}")
    q = lambda s: conn.execute(s).fetchone()[0]
    n_pending = q("SELECT COUNT(*) FROM matches WHERE status = 'pending'")
    print(f"stories: {q('SELECT COUNT(*) FROM stories')} "
          f"({q('SELECT COUNT(*) FROM stories WHERE match_id IS NOT NULL')} with variants) | "
          f"pending: {n_pending}")
    cost = conn.execute("SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM curation_runs").fetchone()
    print(f"tokens so far: {cost[0]} in / {cost[1]} out")


def show_match(conn, mid, kind, conf, summary, rationale, proposal) -> None:
    p = json.loads(proposal)
    print("\n" + "=" * 100)
    print(f"#{mid}  {kind.upper()}  confidence={conf}\n{summary}")
    if kind == "story":
        print(f"shared facts: {rationale}\nroot reason: {p['root_reason']}")
        for bid in p["member_ids"]:
            text, n = conn.execute(
                "SELECT text, (SELECT COUNT(*) FROM bullet_resumes WHERE bullet_id = ?) FROM bullets WHERE id = ?",
                (bid, bid)).fetchone()
            tag = "ROOT" if bid == p["root_id"] else ("NEW " if bid == p.get("added_bullet_id") else "    ")
            print(f"  {tag} #{bid} ({n}x) {text}")
    else:
        print(f"rationale: {rationale}")


def cmd_review(conn, client, args) -> int:
    pending = conn.execute(
        """SELECT id, kind, confidence, summary, rationale, proposal_json FROM matches WHERE status = 'pending'
           ORDER BY CASE kind WHEN 'company' THEN 0 WHEN 'position' THEN 1 ELSE 2 END, id""").fetchall()
    if not pending:
        print("Nothing pending.")
        return 0
    print(f"{len(pending)} pending. Keys: a = accept, r = reject, s = skip, q = quit")
    for row in pending:
        show_match(conn, *row)
        while True:
            try:
                key = input("[a/r/s/q] > ").strip().lower()
            except EOFError:
                key = "q"
            if key in ("a", "r", "s", "q"):
                break
        if key == "q":
            break
        if key in ("a", "r"):
            with conn:
                conn.execute("UPDATE matches SET status = ?, decided_at = datetime('now') WHERE id = ?",
                             ("accepted" if key == "a" else "rejected", row[0]))
    print_status(conn)
    return 0


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------


def _merge_position(conn, src: int, dst: int) -> None:
    conn.execute("UPDATE bullets SET position_id = ? WHERE position_id = ?", (dst, src))
    conn.execute("UPDATE bullet_resumes SET position_id = ? WHERE position_id = ?", (dst, src))
    conn.execute("DELETE FROM positions WHERE id = ?", (src,))
    conn.execute("DELETE FROM position_map WHERE position_id = ?", (src,))


def _rename(conn, table: str, row_id: int, new: str, scope_sql: str = "", scope_args: tuple = ()) -> int:
    """Rename a row; if another row already has that name (case-insensitive) in scope,
    return that row's id so the caller merges into it instead."""
    col = "name" if table == "companies" else "title"
    clash = conn.execute(f"SELECT id FROM {table} WHERE {col} = ? AND id != ? {scope_sql}",
                         (new, row_id, *scope_args)).fetchone()
    if clash:
        return clash[0]
    conn.execute(f"UPDATE {table} SET {col} = ? WHERE id = ?", (new, row_id))
    return row_id


def cmd_apply(conn, client, args) -> int:
    n_pending = conn.execute("SELECT COUNT(*) FROM matches WHERE status = 'pending'").fetchone()[0]
    if n_pending:
        print(f"{n_pending} matches still pending. Run `review` first.")
        return 1
    stamp, n = time.strftime("%Y%m%d_%H%M%S"), 0
    backup = args.db.with_name(f"{args.db.stem}.bak-{stamp}{args.db.suffix}")
    while backup.exists():  # never overwrite an earlier backup
        n += 1
        backup = args.db.with_name(f"{args.db.stem}.bak-{stamp}-{n}{args.db.suffix}")
    shutil.copy2(args.db, backup)
    print(f"Backup: {backup}")
    stats = Counter()
    with conn:
        # 1. Companies (each map row has its own decision: incremental adds join older groups)
        for ccid, cname in conn.execute("SELECT id, name FROM curated_companies").fetchall():
            rows = conn.execute("SELECT company_id, match_id FROM company_map WHERE curated_company_id = ?", (ccid,)).fetchall()
            members = [cid for cid, mid in rows if _accepted(conn, mid)
                       and conn.execute("SELECT 1 FROM companies WHERE id = ?", (cid,)).fetchone()]
            if not members:
                continue
            keep = _rename(conn, "companies", min(members), cname)
            for cid in members:
                if cid == keep:
                    continue
                for pid, title in conn.execute("SELECT id, title FROM positions WHERE company_id = ?", (cid,)).fetchall():
                    twin = conn.execute("SELECT id FROM positions WHERE company_id = ? AND title = ?", (keep, title)).fetchone()
                    if twin:
                        _merge_position(conn, pid, twin[0])
                    else:
                        conn.execute("UPDATE positions SET company_id = ? WHERE id = ?", (keep, pid))
                conn.execute("UPDATE bullets SET company_id = ? WHERE company_id = ?", (keep, cid))
                conn.execute("DELETE FROM companies WHERE id = ?", (cid,))
                conn.execute("DELETE FROM company_map WHERE company_id = ?", (cid,))
                stats["companies merged"] += 1

        # 2. Positions (members may have been merged into a twin in step 1 -> skip vanished ids)
        for cpid, title in conn.execute("SELECT id, title FROM curated_positions").fetchall():
            rows = [r for r in conn.execute("SELECT position_id, match_id FROM position_map WHERE curated_position_id = ?",
                                            (cpid,)).fetchall() if _accepted(conn, r[1])]
            if not rows:
                continue
            live = conn.execute(
                f"SELECT id, company_id FROM positions WHERE id IN ({','.join('?' * len(rows))})", [r[0] for r in rows]).fetchall()
            by_company: dict[int, list[int]] = {}
            for pid, cid in live:
                by_company.setdefault(cid, []).append(pid)
            for cid, pids in by_company.items():   # only merge within the same (final) company
                keep = _rename(conn, "positions", min(pids), title, "AND company_id = ?", (cid,))
                for pid in pids:
                    if pid != keep:
                        _merge_position(conn, pid, keep)
                        stats["positions merged"] += 1

        # 3. Stories
        for (sid,) in conn.execute("SELECT id FROM stories").fetchall():
            stats["bullets linked to a root"] += sync_story(conn, sid)

    problems = conn.execute("PRAGMA foreign_key_check").fetchall()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    q = lambda s: conn.execute(s).fetchone()[0]
    print(f"Now: {q('SELECT COUNT(*) FROM companies')} companies, {q('SELECT COUNT(*) FROM positions')} positions, "
          f"{q('SELECT COUNT(*) FROM bullets WHERE parent_id IS NULL')} root stories, "
          f"{q('SELECT COUNT(*) FROM bullets WHERE parent_id IS NOT NULL')} variants")
    print("Foreign key check:", "clean" if not problems else problems)
    return 0 if not problems else 1


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("entities", cmd_entities), ("stories", cmd_stories), ("add", cmd_add), ("status", None),
                     ("review", cmd_review), ("apply", cmd_apply)]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        if name in ("entities", "stories"):
            p.add_argument("--redo", action="store_true", help="discard previous results for this step and redo")
        if name == "add":
            p.add_argument("resume_json", type=Path, help="A parsed resume JSON (output of parse_resume.py)")
    args = ap.parse_args()
    if not args.db.exists():
        print(f"No database at {args.db}", file=sys.stderr)
        return 2
    conn = setup(args.db)
    if args.func is None:
        print_status(conn)
        return 0
    client = anthropic.Anthropic() if args.cmd in ("entities", "stories", "add") else None
    try:
        return args.func(conn, client, args)
    except anthropic.RateLimitError as e:
        print(f"Rate limited, try again shortly: {e}", file=sys.stderr)
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}", file=sys.stderr)
    except anthropic.APIConnectionError as e:
        print(f"Could not reach the API: {e}", file=sys.stderr)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
    except TypeError as e:
        if "authentication" not in str(e):
            raise
        print('No Anthropic credentials found. Set ANTHROPIC_API_KEY, e.g.\n  $env:ANTHROPIC_API_KEY = "sk-ant-..."', file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
