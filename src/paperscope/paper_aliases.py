import re

PAPER_ALIASES = {
    "corrective rag": "crag_yan_2024",
    "corrective retrieval augmented generation": "crag_yan_2024",
    "rag survey": "rag_survey_gao_2023",
    "self-rag": "self_rag_asai_2023",
    "self rag": "self_rag_asai_2023",
    "crag": "crag_yan_2024",
    "raptor": "raptor_sarthi_2024",
    "dpr": "dpr_karpukhin_2020",
    "colbert": "colbert_khattab_2020",
    "hyde": "hyde_gao_2023",
    "lost in the middle": "lost_in_middle_liu_2023",
    "qwen3 embedding": "qwen3_embedding_2025",
    "original rag": "rag_lewis_2020",
}

def detect_paper_ids(query: str):
    query_lower = query.lower()

    detected = []

    for alias, paper_id in PAPER_ALIASES.items():
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", query_lower):
            detected.append(paper_id)

    return list(dict.fromkeys(detected))