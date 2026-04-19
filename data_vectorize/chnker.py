class Chunker:
    """
    Splits articles into child chunks (for retrieval) with parent context preserved.

    Each entry in `chunked_instances` is a dict:
        {"text": str, "parent_text": str, "section": str}

    "text"       : child chunk (~600 chars), embedded and used for retrieval
    "parent_text": full section text, returned to the LLM for answer generation
    "section"    : section name (e.g. "Results", "Methodology")
    """

    CHILD_MAX_CHARS = 600
    OVERLAP_CHARS = 100

    def __init__(self):
        self.article_id = None
        self.article_title = ""
        self.chunked_instances = []
        self._article_sections = {}

    def set_chunk(self, article_dict: dict):
        self.chunked_instances = []
        self.article_id, article = next(iter(article_dict.items()))
        self.article_title = article.get("Title", str(self.article_id))
        self._article_sections = {k: v for k, v in article.items() if k != "Title"}
        self._split_document()

    def get_chunked_article(self) -> dict:
        return {
            "id": self.article_id,
            "title": self.article_title,
            "chunks": self.chunked_instances,
        }

    def _split_document(self):
        for section_name, content in self._article_sections.items():
            if isinstance(content, list):
                section_text = "\n".join(content)
            elif not isinstance(content, str):
                section_text = str(content)
            else:
                section_text = content

            section_text = section_text.strip()
            if not section_text:
                continue

            for child in self._recursive_split(section_text):
                self.chunked_instances.append({
                    "text": child,
                    "parent_text": section_text,
                    "section": section_name,
                })

    def _recursive_split(self, text: str) -> list[str]:
        """Split text into chunks <= CHILD_MAX_CHARS using progressively finer separators."""
        if len(text) <= self.CHILD_MAX_CHARS:
            return [text]

        for sep in ["\n\n", "\n", ". ", " "]:
            if sep not in text:
                continue
            result = self._split_by_sep(text, sep)
            if result:
                return result

        return self._hard_split(text)

    def _split_by_sep(self, text: str, sep: str) -> list[str]:
        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self.CHILD_MAX_CHARS:
                current = candidate
            else:
                if not current:
                    return []  # single part already exceeds limit; try next separator
                chunks.append(current.strip())
                # Carry the tail of the previous chunk into the next one,
                # snapping forward to a sentence boundary so the overlap is a complete thought.
                tail = current[-self.OVERLAP_CHARS:]
                boundary = tail.find(". ")
                overlap = tail[boundary + 2:] if boundary != -1 else tail
                current = (overlap + sep + part) if overlap else part

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if c]

    def _hard_split(self, text: str) -> list[str]:
        """Last-resort character split when no separator produces small-enough pieces."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.CHILD_MAX_CHARS, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - self.OVERLAP_CHARS
        return chunks
