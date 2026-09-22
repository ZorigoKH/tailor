"""The evidence corpus: your own material, cut into quotable pieces with stable ids.

Everything this package says about you has to come from here, and every claim carries the
id of the piece it came from (``terran.md:14-17``), so a reader can check it. Pieces are
markdown paragraphs and list items; short ones are merged with their neighbour so a citation
is never a fragment. The heading a piece sits under travels with it as its title, because
"Replaced hardcoded business rules" means little without "Terran Enterprise".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

MIN_WORDS = 12  # a piece shorter than this is merged forward
SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class Evidence:
    """One quotable piece of the corpus."""

    id: str
    source: str
    title: str
    text: str
    lines: tuple[int, int]
    tags: tuple[str, ...] = field(default=())

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def quotable(self) -> str:
        """What a model may quote from: the heading context and the text, as one string."""
        return f"{self.title}. {self.text}"

    def cite(self) -> str:
        return f"{self.title} ({self.id})"


def _front_matter(text: str) -> tuple[dict[str, str], str, int]:
    """Read an optional ``---`` header; returns (fields, the rest, lines consumed)."""
    if not text.startswith("---"):
        return {}, text, 0
    close = text.find("\n---", 3)
    if close < 0:
        return {}, text, 0
    head = text[3:close]
    rest = text[close + 4 :]
    if rest.startswith("\n"):
        rest = rest[1:]  # the newline that ends the closing ---
    fields: dict[str, str] = {}
    for line in head.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip().lower()] = value.strip()
    consumed = text[: close + 4].count("\n") + 1
    return fields, rest, consumed


BULLET = re.compile(r"^\s*([-*\u2022]|\d+[.)])\s+")


def chunk_markdown(text: str, source: str, *, title: str = "", tags: tuple = ()) -> list[Evidence]:
    """Cut one document into evidence pieces, keeping 1-based line spans."""
    fields, body, offset = _front_matter(text)
    doc_title = fields.get("title") or title or Path(source).stem.replace("-", " ")
    doc_tags = tuple(t.strip() for t in fields.get("tags", "").split(",") if t.strip()) or tags

    # pass 1: raw blocks - a paragraph, or one list item - with their heading
    heading = ""
    blocks: list[list] = []  # [heading, text, first line, last line]
    current: list | None = None
    for number, raw in enumerate(body.splitlines(), start=offset + 1):
        line = raw.rstrip()
        if not line.strip():
            current = None
            continue
        if line.lstrip().startswith("#"):
            heading = line.lstrip("# ").strip()
            current = None
            continue
        is_item = bool(BULLET.match(line))
        content = BULLET.sub("", line, count=1)
        if current is None or is_item:
            current = [heading, content, number, number]
            blocks.append(current)
        else:
            current[1] = f"{current[1]} {content}"
            current[3] = number

    # pass 2: a short block joins the next block under the same heading, so a citation
    # is never a fragment; a short block at the end of its section joins the previous one
    merged: list[list] = []
    pending: list | None = None
    for block in blocks:
        if pending is not None:
            if block[0] == pending[0]:
                block = [block[0], f"{pending[1]} {block[1]}", pending[2], block[3]]
            elif merged and merged[-1][0] == pending[0]:
                merged[-1] = [
                    pending[0],
                    f"{merged[-1][1]} {pending[1]}",
                    merged[-1][2],
                    pending[3],
                ]
            else:
                merged.append(pending)
            pending = None
        if len(block[1].split()) < MIN_WORDS:
            pending = block
        else:
            merged.append(block)
    if pending is not None:
        if merged and merged[-1][0] == pending[0]:
            merged[-1] = [pending[0], f"{merged[-1][1]} {pending[1]}", merged[-1][2], pending[3]]
        else:
            merged.append(pending)

    pieces = []
    for head, text_, first, last in merged:
        if not text_.strip():
            continue
        if not head or head.lower() in doc_title.lower():
            full_title = doc_title
        elif doc_title.lower() in head.lower():
            full_title = head  # the heading already names the document
        else:
            full_title = f"{doc_title} - {head}"
        pieces.append(
            Evidence(
                id=f"{source}:{first}-{last}",
                source=source,
                title=full_title,
                text=text_.strip(),
                lines=(first, last),
                tags=doc_tags,
            )
        )
    return pieces


def load_corpus(path: str | os.PathLike) -> list[Evidence]:
    """Load every ``.md``/``.txt`` file under ``path`` as evidence, sorted by file name."""
    root = Path(path)
    if root.is_file():
        files = [root]
        root = root.parent
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no .md or .txt files in {path}")
    pieces: list[Evidence] = []
    for file in files:
        name = str(file.relative_to(root))
        pieces += chunk_markdown(file.read_text(encoding="utf-8"), name)
    if not pieces:
        raise ValueError(f"{path} has files but no readable text in them")
    return pieces


def by_id(pieces: list[Evidence]) -> dict[str, Evidence]:
    return {piece.id: piece for piece in pieces}
