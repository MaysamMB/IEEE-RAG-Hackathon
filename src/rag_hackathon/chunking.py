from typing import Iterable

def first_whitespace(text: str, positions: Iterable[int]) -> int:
    return next((position for position in positions if text[position].isspace()), -1)

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
        """Split at whitespace so words stay whole; only a single word longer than size is cut."""
        parts: list[str] = []
        start = 0
        while True:
            end = start + size
            if end >= len(paragraph):
                parts.append(paragraph[start:].strip())
                return parts
            space = first_whitespace(paragraph, range(end, start, -1))
            if space == -1:
                parts.append(paragraph[start:end].strip())
                start = end - overlap
                continue
            parts.append(paragraph[start:space].strip())
            next_word = first_whitespace(paragraph, range(max(start + 1, space - overlap), space))
            start = (space if next_word == -1 else next_word) + 1

    def overlap_tail(chunk: str) -> str:
        """Return the overlap text, dropping a partial word at its start."""
        tail = chunk[-overlap:]
        if len(chunk) > overlap and not chunk[-overlap - 1].isspace():
            whole_words = tail.split(maxsplit=1)
            tail = whole_words[1] if len(whole_words) > 1 else ""
        return tail

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
            tail = overlap_tail(current) if overlap and current else ""
            current = f"{tail}\n{paragraph}".strip() if tail else paragraph
            if len(current) > size:
                chunks.extend(part for part in split_long(current) if part)
                current = ""
    if current:
        chunks.append(current)
    return chunks

def make_chunk_id(metadata: dict) -> str:
    """Build a deterministic ID; source_path keeps same-named files in different folders apart."""
    return f"{metadata['source_path']}::p{metadata['page']}::c{metadata['chunk_index']}"

def chunk_records(records: Iterable[dict], size: int, overlap: int) -> list[dict]:
    output: list[dict] = []
    for record in records:
        for index, chunk in enumerate(chunk_text(record["text"], size, overlap)):
            metadata = {**record["metadata"], "chunk_index": index}
            metadata["chunk_id"] = make_chunk_id(metadata)
            output.append({"text": chunk, "metadata": metadata})
    return output
