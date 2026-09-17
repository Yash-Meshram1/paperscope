import requests


def build_context(results):
    context_parts = []

    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]

        context_parts.append(
            f"""
            [SOURCE {i}]
            Paper: {chunk['title']}
            Section: {chunk['section_heading']}
            Pages: {chunk['page_start']}-{chunk['page_end']}

            {chunk['text']}
            """.strip()
        )

    return "\n\n".join(context_parts)


def build_prompt(query, results):
    context = build_context(results)

    return f"""
You are a research assistant answering questions about a collection of research papers.

Use only the evidence provided below.

Rules:
- Do not use outside knowledge.
- Every factual claim should be supported by the supplied evidence.
- Cite sources using [SOURCE 1], [SOURCE 2], etc.
- Prefer primary-source evidence over papers that merely discuss another work.
- Use the minimum number of citations needed to support each claim.
- If the evidence is insufficient, explicitly say that the provided papers do not contain enough information.
- Do not invent citations.

Question:
{query}

Evidence:
{context}

Answer:
""".strip()


def generate_answer(
    query,
    results,
    model="qwen3.5:9b",
    temperature=0.1,
    max_tokens=800,
    ollama_url="http://localhost:11434/api/chat"
):
    prompt = build_prompt(query, results)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    response = requests.post(
        ollama_url,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    return result["message"]["content"]