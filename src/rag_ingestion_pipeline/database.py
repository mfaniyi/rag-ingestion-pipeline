from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


# Define the PostgreSQL connection URL for the Docker database.
DATABASE_URL = (
    "postgresql+psycopg://raguser:ragpassword@localhost:5433/ragdb"
)


# Create the SQLAlchemy database engine.
engine = create_engine(DATABASE_URL)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy database models."""

    pass


class DocumentChunk(Base):
    """Store document chunks and their vector embeddings."""

    __tablename__ = "document_chunks"

    # Store a unique identifier for each chunk.
    chunk_id: Mapped[str] = mapped_column(primary_key=True)

    # Store the ID of the source document.
    document_id: Mapped[str]

    # Store the actual chunk text.
    text: Mapped[str]

    # Store the chunk's position within the document.
    chunk_index: Mapped[int]

    # Store the source page number when available.
    page_number: Mapped[int | None]

    # Store the chunking strategy used to create the chunk.
    strategy: Mapped[str]

    # Store the 384-dimensional embedding vector.
    embedding: Mapped[list[float]] = mapped_column(Vector(384))


def create_tables() -> None:
    """Create the database tables."""

    # Create all tables defined by the SQLAlchemy models.
    Base.metadata.create_all(engine)


def save_chunk(chunk, embedding: list[float]) -> None:
    """Save one document chunk and its embedding to PostgreSQL."""

    # Create a database session for the insert operation.
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        # Create a database record from the chunk metadata and embedding.
        record = DocumentChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            strategy=chunk.strategy,
            embedding=embedding,
        )

        # Add the record to the current database transaction.
        session.add(record)

        # Commit the transaction to permanently store the chunk.
        session.commit()


def save_chunks(chunks, embeddings: list[list[float]]) -> None:
    """Save multiple document chunks and their embeddings."""

    from sqlalchemy.orm import Session

    # Create a database session for the bulk insert.
    with Session(engine) as session:
        # Convert each chunk and embedding into a database record.
        records = [
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                strategy=chunk.strategy,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Add all records to the current database transaction.
        session.add_all(records)

        # Commit all chunks as one database transaction.
        session.commit()