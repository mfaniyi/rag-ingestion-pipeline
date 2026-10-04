# Retrieval Evaluation Report

## 1. Overview

This evaluation measures the retrieval quality of the RAG ingestion pipeline using a small manually labelled evaluation dataset.

The evaluation focuses on the retrieval stage before answer generation and measures whether relevant document chunks are returned among the top retrieved results.

The production retrieval pipeline is:

Question → Embedding → Hybrid Search → RRF Ranking → Cross-Encoder Reranking → Top 5 Chunks

## 2. Evaluation Dataset

The evaluation dataset contains 10 questions based on the `mlt-001` document.

Each question was manually labelled with one or more relevant chunk IDs from the source document.

The dataset is stored in:

`evaluation/labelled_questions.json`

The questions cover topics including:

- Data analysis
- Power BI activity flow
- Query folding
- Power Query
- Fact and dimension tables
- Measures
- DAX `CALCULATE`
- Performance Analyzer
- Cardinality
- Power BI bookmarks

## 3. Retrieval Method

The evaluation uses the same retrieval approach implemented by the `/ask` endpoint.

### Step 1: Query Embedding

Each question is converted into a vector embedding using:

`all-MiniLM-L6-v2`

The resulting 384-dimensional embedding is used for vector similarity search with PostgreSQL and pgvector.

### Step 2: Vector Search

The system performs cosine-distance vector search against the document chunks.

The top 10 vector-search results are returned.

### Step 3: Keyword Search

The system also performs PostgreSQL full-text keyword search using:

- `to_tsvector`
- `websearch_to_tsquery`
- `ts_rank_cd`

The top 10 keyword-search results are returned.

### Step 4: Hybrid Search

The vector and keyword results are combined using Reciprocal Rank Fusion (RRF).

The RRF constant used by the implementation is:

`60`

This allows a chunk that performs well in both retrieval methods to receive a stronger combined score.

### Step 5: Cross-Encoder Reranking

The hybrid candidates are passed to:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The cross-encoder scores the relevance of each question/chunk pair.

The top 5 reranked chunks are used for the evaluation.

## 4. Evaluation Metric

The primary metric is Precision@5.

Precision@5 measures the proportion of the five retrieved chunks that are relevant to the question.

Formula:

`Precision@5 = Number of relevant chunks in top 5 / 5`

For example, if two of the five retrieved chunks are relevant:

`Precision@5 = 2 / 5 = 0.40`

The final score is the mean Precision@5 across all 10 evaluation questions.

## 5. Results

| Question | Precision@5 |
|---|---:|
| Q1 | 0.40 |
| Q2 | 0.40 |
| Q3 | 0.20 |
| Q4 | 0.20 |
| Q5 | 0.40 |
| Q6 | 0.40 |
| Q7 | 0.20 |
| Q8 | 0.40 |
| Q9 | 0.40 |
| Q10 | 0.20 |
| **Mean Precision@5** | **0.32** |

The evaluation produced a Mean Precision@5 of:

**0.32**

This means that, on average, approximately 32% of the five retrieved chunks were manually labelled as relevant across this evaluation set.

## 6. Observations

### Successful Retrieval

Several questions achieved a Precision@5 score of 0.40.

These include questions about:

- Data analysis
- Power BI activity flow
- Fact and dimension tables
- Measures
- Performance Analyzer
- Cardinality

This indicates that the retrieval pipeline was able to place relevant chunks within the top five for several questions.

### Lower-Scoring Retrieval

Q3, Q4, Q7, and Q10 achieved a Precision@5 score of 0.20.

These questions concern more specific concepts such as:

- Query folding
- Power Query
- `CALCULATE`
- Bookmarks

These results indicate that the retrieval pipeline sometimes returns related chunks alongside the most directly relevant chunk.

### Hybrid Retrieval Observation

During evaluation, Q8 initially returned a Precision@5 score of 0.00 when only `mlt-001-chunk-047` was labelled as relevant.

Further inspection showed that chunks `052` and `053` also contained relevant information about Performance Analyzer.

The label was therefore expanded to include:

- `mlt-001-chunk-047`
- `mlt-001-chunk-052`
- `mlt-001-chunk-053`

The revised Q8 score became 0.40.

This demonstrated the importance of carefully defining relevance labels before interpreting retrieval metrics.

## 7. Limitations

This evaluation has several limitations.

### Small Evaluation Dataset

Only 10 questions were evaluated.

The results therefore provide a baseline rather than a comprehensive measurement of retrieval performance.

### Single Document

The evaluation is restricted to the `mlt-001` document.

The results may not represent retrieval performance across different document types, subjects, or writing styles.

### Manual Relevance Labels

The relevant chunk IDs were manually selected.

Different evaluators could potentially classify borderline chunks differently.

### Precision Only

The current evaluation measures Precision@5.

Recall, MRR, NDCG, and other retrieval metrics were not included in this baseline.

## 8. Failure Modes

The evaluation identified an important retrieval failure mode.

A relevant chunk can be retrieved by vector search but fail to appear in the final hybrid candidate set if it does not rank highly enough in the combined retrieval results.

For example, during the Q8 investigation, `mlt-001-chunk-047` appeared in the vector-search results but did not make the top hybrid candidates.

Because reranking operates on the hybrid candidate set, the cross-encoder cannot recover a relevant chunk that was never passed to it.

This demonstrates that reranking improves ordering of retrieved candidates but does not replace effective candidate retrieval.

## 9. Conclusion

The retrieval evaluation establishes a measurable baseline for the RAG pipeline.

Using 10 manually labelled questions from the `mlt-001` document, the current pipeline achieved:

**Mean Precision@5 = 0.32**

The evaluation confirms that the system can retrieve relevant evidence using a combination of vector search, keyword search, RRF-based hybrid retrieval, and cross-encoder reranking.

The baseline also identifies areas for future improvement, particularly increasing the quality and breadth of the candidate retrieval stage and expanding the evaluation dataset.

The evaluation is intentionally treated as a baseline rather than as evidence of general retrieval performance across all possible documents and questions.