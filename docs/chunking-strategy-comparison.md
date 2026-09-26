# Chunking Strategy Comparison

## Objective

The objective of this experiment was to implement and compare two document chunking strategies using the same PDF document:

1. Fixed-size chunking with overlap
2. Paragraph-based chunking

The purpose was to determine which strategy produces more useful chunks for retrieval in a RAG ingestion pipeline.

---

## Test Document

**Document:** `MLT.pdf`

The document was processed using the ingestion pipeline, which extracted the PDF text while preserving page numbers.

---

## Strategy 1: Fixed-Size Chunking

The first strategy divides the document into chunks with a maximum size of 500 characters.

A 50-character overlap was used between consecutive chunks to help preserve context across chunk boundaries.

### Configuration

- Chunk size: 500 characters
- Overlap: 50 characters
- Strategy: `fixed_size`

### Results

- Number of chunks: 101
- Minimum chunk size: 20 characters
- Maximum chunk size: 500 characters
- Average chunk size: 438.75 characters

The minimum chunk size occurred because the final portion of some pages contained fewer than 500 remaining characters.

---

## Strategy 2: Paragraph-Based Chunking

The second strategy splits the document using paragraph boundaries.

This approach attempts to preserve complete blocks of related text instead of splitting text at a fixed character position.

### Configuration

- Strategy: `paragraph`
- Overlap: None

### Results

- Number of chunks: 23
- Minimum chunk size: 1,019 characters
- Maximum chunk size: 2,762 characters
- Average chunk size: 1,755.65 characters

---

## Comparison

| Metric | Fixed-size | Paragraph |
|---|---:|---:|
| Number of chunks | 101 | 23 |
| Minimum characters | 20 | 1,019 |
| Maximum characters | 500 | 2,762 |
| Average characters | 438.75 | 1,755.65 |
| Overlap | 50 characters | None |

---

## Observations

### Fixed-size chunking

Fixed-size chunking produced considerably more chunks with relatively consistent sizes.

The 50-character overlap helps maintain some context between adjacent chunks. However, because the split is based on character positions, a sentence or word can be divided between two chunks.

For example, one chunk ended with:

> `...assessment oppor`

while the following chunk began with:

> `igned to supplement...`

This demonstrates a limitation of fixed-size chunking.

### Paragraph-based chunking

Paragraph-based chunking produced fewer but substantially larger chunks.

The largest chunk contained 2,762 characters and included multiple questions and answers. While this approach preserves larger sections of context, large chunks may contain several different concepts in the same retrieval unit.

---

## Final Decision

For this project, **fixed-size chunking with 500-character chunks and 50-character overlap** was selected as the primary chunking strategy.

The decision was based on the following considerations:

1. The chunks have more predictable sizes.
2. Smaller chunks can provide more specific retrieval units.
3. The overlap helps preserve context between adjacent chunks.
4. The paragraph strategy produced significantly larger chunks.
5. Some paragraph chunks contained multiple questions or concepts, which may reduce retrieval precision.

### Trade-off

Fixed-size chunking can split sentences or ideas across chunk boundaries. The 50-character overlap reduces this problem but does not eliminate it completely.

Paragraph-based chunking preserves larger semantic sections but can produce chunks that are too large for precise retrieval.

Therefore, fixed-size chunking was selected as the initial strategy for the RAG ingestion pipeline, while recognizing that a more advanced semantic chunking approach could be evaluated in a future iteration.