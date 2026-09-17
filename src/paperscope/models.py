from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Section:
    heading: str
    section_number: Optional[str]
    level: int
    page_start: Optional[int]
    page_end: Optional[int]
    text: str


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: List[str]
    year: int
    source_file: str
    sections: List[Section] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    section_heading: str
    section_number: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_index: int
    text: str