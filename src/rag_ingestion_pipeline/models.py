from dataclasses import dataclass


@dataclass
class Document:
    document_id: str
    filename: str
    content: str


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    page_number: int | None
    strategy: str
