"""File sources: where the raw resume files come from.

LocalFolderSource is the only implementation today. The abstract base
class is the seam for a future cloud-storage source (S3, Google Drive,
etc.) -- it only needs to list identifiers and open a readable binary
stream for each one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

DEFAULT_EXTENSIONS = (".docx",)


class FileSource(ABC):
    @abstractmethod
    def list_files(self) -> list[str]:
        """Return identifiers for every readable file in this source."""

    @abstractmethod
    def open(self, identifier: str) -> BinaryIO:
        """Open the file for the given identifier as a binary stream."""


class LocalFolderSource(FileSource):
    def __init__(self, folder: str | Path, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise ValueError(f"Not a directory: {self.folder}")
        self.extensions = tuple(ext.lower() for ext in extensions)

    def list_files(self) -> list[str]:
        return sorted(
            str(path)
            for path in self.folder.rglob("*")
            if path.is_file() and path.suffix.lower() in self.extensions
        )

    def open(self, identifier: str) -> BinaryIO:
        return open(identifier, "rb")
