import re
from .models import Paper, Section


def get_section_level(heading: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)\s+", heading)

    if not match:
        return 1

    section_number = match.group(1)
    return section_number.count(".") + 1


def get_section_number(heading: str):
    match = re.match(r"^(\d+(?:\.\d+)*)\s+", heading)

    if match:
        return match.group(1)

    return None


def build_sections(doc):
    sections = []
    current_section = None

    for item, level in doc.iterate_items():
        item_type = type(item).__name__
        text = getattr(item, "text", "").strip()

        if not text:
            continue

        prov = getattr(item, "prov", [])
        page_no = prov[0].page_no if prov else None

        if "SectionHeader" in item_type:
            current_section = {
                "heading": text,
                "section_number": get_section_number(text),
                "level": get_section_level(text),
                "pages": [],
                "text_parts": []
            }

            if page_no is not None:
                current_section["pages"].append(page_no)

            sections.append(current_section)

        elif current_section is not None:
            current_section["text_parts"].append(text)

            if page_no is not None:
                current_section["pages"].append(page_no)

    section_objects = []

    for section in sections:
        text = "\n\n".join(section["text_parts"])

        page_start = min(section["pages"]) if section["pages"] else None
        page_end = max(section["pages"]) if section["pages"] else None

        section_objects.append(
            Section(
                heading=section["heading"],
                section_number=section["section_number"],
                level=section["level"],
                page_start=page_start,
                page_end=page_end,
                text=text
            )
        )

    return section_objects

def build_paper(
    doc,
    paper_id: str,
    title: str,
    authors: list[str],
    year: int,
    source_file: str
):
    sections = build_sections(doc)

    return Paper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        source_file=source_file,
        sections=sections
    )