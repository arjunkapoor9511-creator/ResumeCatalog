# ResumeCatalog
manages your resume

## Bullet/paragraph catalog extractor

Scans a folder of `.docx` resume files (anything else, including PDFs, is
ignored), splits each one into bullet points and paragraph blocks, and
writes a deduplicated catalog to JSON. Chunks are deduped by exact match
after whitespace/case normalization — anything that differs even slightly
is logged as a distinct entry.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py --input path\to\resumes --output catalog.json -v
```

The input folder is searched recursively (nested subfolders included).

To only include files whose name matches one or more patterns (e.g. only
files with "resume" or "cv" in the name), pass `--name-pattern` one or more
times. Each pattern is a regex matched case-insensitively anywhere in the
filename; a file is included if it matches *any* of them. If no
`--name-pattern` is given, every `.docx` file is included.

```
.venv\Scripts\python main.py --input path\to\resumes --output catalog.json --name-pattern resume --name-pattern cv
```

Pass `--bullets-only` to write only bullet-point chunks to the catalog,
skipping plain paragraph chunks (headers, narrative text, etc.) entirely.

Each catalog entry records the chunk text, whether it's a `"bullet"` or a
`"paragraph"`, and the name of the file it was first seen in (just the
filename, not the full path).

If a `.docx` has Track Changes, text is read as Word's "No Markup" view
would show it: accepted insertions are included, proposed deletions are
excluded -- regardless of whether the changes have actually been accepted
in the file.

Input is read through `resume_catalog.sources.FileSource`
([resume_catalog/sources.py](resume_catalog/sources.py)), currently
implemented only for a local folder (`LocalFolderSource`). A future
cloud-storage source (S3, Google Drive, etc.) just needs to implement
`list_files()` and `open()` against that same interface.
