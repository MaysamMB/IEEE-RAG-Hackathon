from typing import Iterable

def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Paragraph-aware overlapping chunks; sizes are character-based starter values."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []

    paragraphs = normalized.split("\n")
    chunks: list[str] = []
    current = ""

    def split_long(paragraph: str) -> list[str]:
        step = size - overlap
        return [paragraph[i:i + size].strip() for i in range(0, len(paragraph), step)]

    for paragraph in paragraphs:
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(part for part in split_long(paragraph) if part)
            continue
        candidate = f"{current}\n{paragraph}" if current else paragraph
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n{paragraph}".strip() if tail else paragraph
            if len(current) > size:
                chunks.extend(part for part in split_long(current) if part)
                current = ""
    if current:
        chunks.append(current)
    return chunks

def chunk_records(records: Iterable[dict], size: int, overlap: int) -> list[dict]:
    output: list[dict] = []
    for record in records:
        for index, chunk in enumerate(chunk_text(record["text"], size, overlap)):
            output.append({
                "text": chunk,
                "metadata": {**record["metadata"], "chunk_index": index},
            })
    return output
