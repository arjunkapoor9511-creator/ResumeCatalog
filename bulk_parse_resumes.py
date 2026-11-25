"""Bulk-parse every .docx resume in a folder with one Message Batches request.

Files with identical extracted text are parsed once (shortest file name kept).

Usage:
    python bulk_parse_resumes.py <input_folder> <output_folder>
    python bulk_parse_resumes.py <input_folder> <output_folder> --batch-id msgbatch_...   # resume waiting

Batches run asynchronously at 50% of the normal token price; most finish within
an hour (max 24h). The script submits, polls until the batch ends, then writes
<resume name>.json for each success into the output folder.

A manifest (batch_<id>.manifest.json) is written to the output folder right after
submission. If the script is interrupted, rerun with --batch-id to pick up the
results instead of paying for a second batch.

Reuses the schema, prompt, and .docx extraction from parse_resume.py, with
structure enforced the same way: output_config.format with the JSON schema.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import ValidationError

from parse_resume import MODEL, SYSTEM_PROMPT, ParsedResume, blanks_to_null, docx_to_text

OUTPUT_SCHEMA = anthropic.transform_schema(ParsedResume)


def unique_resumes(files: list[Path]) -> list[tuple[Path, str]]:
    """Extract each file's text and skip files whose extracted text is identical
    to another's (what Claude would see is the same, even if the .docx bytes
    differ, e.g. after a re-save). Of each identical group, keep the shortest
    name (e.g. 'CV.docx' over 'CV (1).docx'). Files with no text are skipped."""
    groups: dict[str, list[Path]] = {}
    for path in files:
        text = docx_to_text(path)
        if not text.strip():
            print(f"SKIP  {path.name}: no text found")
            continue
        groups.setdefault(text, []).append(path)
    kept: list[tuple[Path, str]] = []
    for text, group in groups.items():
        group.sort(key=lambda p: (len(p.name), p.name))
        kept.append((group[0], text))
        for dup in group[1:]:
            print(f"DUP   {dup.name}  (same text as {group[0].name})")
    return sorted(kept)


def build_requests(resumes: list[tuple[Path, str]]) -> tuple[list[Request], dict[str, str]]:
    """custom_id must match ^[a-zA-Z0-9_-]{1,64}$, and resume file names have
    spaces and parentheses, so use index ids and keep a manifest back to files."""
    requests: list[Request] = []
    manifest: dict[str, str] = {}
    for i, (path, text) in enumerate(resumes):
        custom_id = f"resume-{i:04d}"
        manifest[custom_id] = str(path)
        requests.append(
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": f"<resume>\n{text}\n</resume>"}],
                    output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                ),
            )
        )
    return requests, manifest


def wait_for(client: anthropic.Anthropic, batch_id: str, poll_seconds: int):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(
            f"[{time.strftime('%H:%M:%S')}] {batch.processing_status}: "
            f"{c.processing} processing, {c.succeeded} succeeded, {c.errored} errored"
        )
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_seconds)


def collect(client: anthropic.Anthropic, batch_id: str, manifest: dict[str, str], out_dir: Path) -> int:
    ok, failures = 0, []
    # Results arrive in any order: key by custom_id, never by position.
    for result in client.messages.batches.results(batch_id):
        source = Path(manifest.get(result.custom_id, result.custom_id))
        r = result.result
        if r.type != "succeeded":
            reason = r.type
            if r.type == "errored":
                reason = f"errored ({r.error.error.type}: {r.error.error.message})"
            failures.append((source.name, reason))
            continue

        msg = r.message
        if msg.stop_reason == "refusal":
            failures.append((source.name, "model declined"))
            continue
        if msg.stop_reason == "max_tokens":
            failures.append((source.name, "hit max_tokens before JSON was complete"))
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            parsed = ParsedResume.model_validate_json(text)
        except ValidationError as e:
            failures.append((source.name, f"output failed validation: {e.error_count()} errors"))
            continue

        out = out_dir / f"{source.stem}.json"
        out.write_text(json.dumps(blanks_to_null(parsed.model_dump()), indent=2, ensure_ascii=False), encoding="utf-8")
        ok += 1
        print(f"OK    {out.name}")

    print(f"\nDone: {ok} written to {out_dir}, {len(failures)} failed.")
    for name, reason in failures:
        print(f"  FAIL {name}: {reason}")
    return 0 if not failures else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_folder", type=Path, help="Folder containing .docx resumes")
    ap.add_argument("output_folder", type=Path, help="Folder to write JSON results into (created if missing)")
    ap.add_argument("--batch-id", help="Resume an already-submitted batch instead of creating a new one")
    ap.add_argument("--poll", type=int, default=60, help="Seconds between status checks (default 60)")
    args = ap.parse_args()

    if not args.input_folder.is_dir():
        print(f"Input folder not found: {args.input_folder}", file=sys.stderr)
        return 2
    if args.output_folder.exists() and not args.output_folder.is_dir():
        print(f"Output path is an existing file: {args.output_folder}", file=sys.stderr)
        return 2
    args.output_folder.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    try:
        if args.batch_id:
            manifest_path = args.output_folder / f"batch_{args.batch_id}.manifest.json"
            if not manifest_path.exists():
                print(f"No manifest at {manifest_path}; results will be named by request id.", file=sys.stderr)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            batch_id = args.batch_id
        else:
            files = sorted(p for p in args.input_folder.glob("*.docx") if not p.name.startswith("~$"))
            if not files:
                print(f"No .docx files in {args.input_folder}", file=sys.stderr)
                return 1
            unique = unique_resumes(files)
            if len(unique) < len(files):
                print(f"\n{len(files)} files -> {len(unique)} unique resumes to parse.\n")
            requests, manifest = build_requests(unique)
            if not requests:
                print("Nothing to submit.", file=sys.stderr)
                return 1
            batch = client.messages.batches.create(requests=requests)
            batch_id = batch.id
            manifest_path = args.output_folder / f"batch_{batch_id}.manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Submitted batch {batch_id} with {len(requests)} resumes.")
            print(f"If interrupted, resume with: --batch-id {batch_id}\n")

        wait_for(client, batch_id, args.poll)
        return collect(client, batch_id, manifest, args.output_folder)

    except KeyboardInterrupt:
        print("\nStopped waiting. The batch keeps running; rerun with --batch-id to collect results.")
        return 130
    except anthropic.RateLimitError as e:
        print(f"Rate limited, try again shortly: {e}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as e:
        print(f"Could not reach the API: {e}", file=sys.stderr)
        return 1
    except TypeError as e:
        if "authentication" not in str(e):
            raise
        print('No Anthropic credentials found. Set ANTHROPIC_API_KEY, e.g.\n  $env:ANTHROPIC_API_KEY = "sk-ant-..."', file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
