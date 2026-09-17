from .models import Paper, Section, Chunk

import re


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def split_text_with_overlap(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100
):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks

def chunk_section(
    paper: Paper,
    section: Section,
    section_index: int,
    chunk_size: int = 600,
    overlap: int = 100
):
    text_chunks = split_text_with_overlap(
        section.text,
        chunk_size=chunk_size,
        overlap=overlap
    )

    chunks = []
    section_slug = slugify(section.heading)

    for index, chunk_text in enumerate(text_chunks):
        chunk = Chunk(
            chunk_id=(
                f"{paper.paper_id}_"
                f"s{section_index}_"
                f"{section_slug}_"
                f"{index}"
            ),
            paper_id=paper.paper_id,
            title=paper.title,
            section_heading=section.heading,
            section_number=section.section_number,
            page_start=section.page_start,
            page_end=section.page_end,
            chunk_index=index,
            text=chunk_text
        )

        chunks.append(chunk)

    return chunks