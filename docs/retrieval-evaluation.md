# Retrieval Evaluation

## Evaluation Method

The retrieval pipeline was evaluated using five labelled questions based on the Dangote Petroleum Refinery FAQ document.

For each question, one chunk containing the expected answer was labelled as relevant.

The evaluation compared four retrieval approaches:

1. Vector search
2. Keyword search
3. Hybrid search
4. Hybrid search with cross-encoder reranking

The evaluation metric used was Precision@5.

Precision@5 measures the proportion of the top five retrieved chunks that are relevant.

## Results

| Retrieval Method | Average Precision@5 |
|---|---:|
| Vector Search | 0.20 |
| Keyword Search | 0.16 |
| Hybrid Search | 0.20 |
| Hybrid + Reranking | 0.20 |

## Ranking Results

| Question | Vector | Keyword | Hybrid | Hybrid + Reranking |
|---|---:|---:|---:|---:|
| What is the refinery's processing capacity? | #2 | Not retrieved | #4 | #5 |
| What are DPRP's expansion plans? | #1 | #2 | #1 | #1 |
| Where does DPRP get its crude oil? | #4 | #1 | #2 | #1 |
| What is the Nelson Complexity Index of the refinery? | #1 | #1 | #1 | #1 |
| What products does DPRP make? | #1 | #3 | #2 | #1 |

## Findings

Vector search retrieved the relevant chunk within the top five for all five questions.

Keyword search retrieved the relevant chunk for four of the five questions. It failed to retrieve the labelled processing-capacity chunk within the top five.

Hybrid search also retrieved the relevant chunk for all five questions. However, combining keyword and vector retrieval did not increase the measured Precision@5 compared with vector search in this small evaluation set.

The cross-encoder reranker improved the ranking of some queries. For example, the crude-oil question moved from position 2 with hybrid search to position 1 after reranking.

However, reranking also reduced the ranking of the processing-capacity answer from position 4 to position 5. This demonstrates that reranking does not automatically improve every query.

## Conclusion

The evaluation shows that vector, hybrid, and hybrid-plus-reranking retrieval all achieved an average Precision@5 of 0.20 on this small labelled dataset, while keyword search achieved 0.16.

The results demonstrate that different retrieval methods behave differently depending on the question. Hybrid retrieval and reranking therefore need to be evaluated rather than assumed to improve retrieval quality.

The evaluation set is intentionally small and is suitable as a basic assignment-level retrieval evaluation. A larger labelled dataset would provide a more reliable measurement of retrieval performance.