from pathlib import Path
from fastapi import UploadFile
from io import BytesIO
from pypdf import PdfReader


ALLOWED_EXTENSIONS = {".pdf", ".txt"}

MAX_FILE_SIZE = 10 * 1024 * 1024


def validate_upload(file: UploadFile, file_size: int) -> None:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File exceeds the maximum allowed size of 10 MB")


def extract_text_from_txt(content: bytes) -> str:
    return content.decode("utf-8")


def extract_text_from_pdf(content: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(BytesIO(content))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_number, text))
    return pages


def extract_document(content: bytes, filename: str) -> list[tuple[int | None, str]]:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return extract_text_from_pdf(content)

    if extension == ".txt":
        text = extract_text_from_txt(content)
        return [(None, text)]

    raise ValueError(f"Unsupported file type: {extension}")
