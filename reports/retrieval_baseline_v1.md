# CiteQuest Retrieval Benchmark v1

## Benchmark setup

- Corpus: `arxiv_cs`
- Papers: 50000
- Chunks: 50000
- Queries: 150 (dev=50, test=100)
- Query types: `{"keyword": 50, "natural_question": 50, "semantic_paraphrase": 50}`
- Embedding model: `BAAI/bge-m3`
- FAISS: `IndexIVFFlat`, nlist=894, nprobe=64
- Raw SHA256: `d43205c325be9f6e1d3112277324b59e397ff96626451559db0f05e81ccd6ea0`
- Eval SHA256: `bb5e705bddeb4cab32488b8b5274ebce78a6f6eca9e885be4f816022eb1837a7`
- Git commit: `45d3c0a7973c83ca2d9a0a9bc039d4438f7c8adf`
- Environment: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`

## Metric definitions

- HitRate@K: fraction of queries with at least one relevant paper in top K.
- Recall@K: fraction of all known relevant papers retrieved in top K.
- MRR@10: mean reciprocal rank of the first relevant paper within top 10.
- nDCG@10: binary normalized discounted gain using all known relevant papers.
- Latency: one untimed warm-up per method, then per-query wall-clock time.

## Dev alpha sweep

| Alpha (lexical) | HitRate@10 | Recall@10 | MRR@10 | nDCG@10 | mean ms | p50 ms | p95 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.9200 | 0.9200 | 0.8357 | 0.8556 | 303.646 | 267.352 | 494.907 |
| 0.35 | 0.9200 | 0.9200 | 0.8650 | 0.8784 | 206.569 | 222.966 | 261.912 |
| 0.50 | 0.9200 | 0.9200 | 0.8673 | 0.8804 | 208.115 | 214.782 | 286.653 |
| 0.65 | 0.9400 | 0.9400 | 0.8323 | 0.8578 | 209.613 | 211.609 | 283.345 |
| 0.80 | 0.9400 | 0.9400 | 0.8247 | 0.8514 | 194.553 | 209.278 | 257.836 |

Selected alpha: **0.50**, using dev nDCG@10 with dev MRR@10 as tie-break.

## Final test results

| Method | HitRate@5 | HitRate@10 | Recall@10 | MRR@10 | nDCG@10 | mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.8700 | 0.9000 | 0.9000 | 0.8093 | 0.8315 | 141.609 | 162.003 | 190.497 |
| Dense | 0.8900 | 0.9100 | 0.9100 | 0.7951 | 0.8233 | 23.445 | 21.237 | 27.287 |
| Hybrid 0.5 | 0.9400 | 0.9500 | 0.9500 | 0.8673 | 0.8881 | 196.754 | 203.049 | 263.514 |
| Hybrid tuned (0.50) | 0.9400 | 0.9500 | 0.9500 | 0.8673 | 0.8881 | 196.754 | 203.049 | 263.514 |

## Results by query type

| Method | Query type | N | HitRate@10 | MRR@10 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| BM25 | keyword | 33 | 0.9394 | 0.8351 | 0.8601 |
| BM25 | natural_question | 33 | 1.0000 | 0.9798 | 0.9848 |
| BM25 | semantic_paraphrase | 34 | 0.7647 | 0.6189 | 0.6550 |
| Dense | keyword | 33 | 0.9394 | 0.8093 | 0.8415 |
| Dense | natural_question | 33 | 0.9697 | 0.8990 | 0.9170 |
| Dense | semantic_paraphrase | 34 | 0.8235 | 0.6804 | 0.7147 |
| Hybrid 0.5 | keyword | 33 | 1.0000 | 0.8798 | 0.9103 |
| Hybrid 0.5 | natural_question | 33 | 1.0000 | 0.9848 | 0.9888 |
| Hybrid 0.5 | semantic_paraphrase | 34 | 0.8529 | 0.7412 | 0.7687 |
| Hybrid tuned (0.50) | keyword | 33 | 1.0000 | 0.8798 | 0.9103 |
| Hybrid tuned (0.50) | natural_question | 33 | 1.0000 | 0.9848 | 0.9888 |
| Hybrid tuned (0.50) | semantic_paraphrase | 34 | 0.8529 | 0.7412 | 0.7687 |

## Error analysis

### `hybrid_success_bm25_failure` (5)

- `q0082` (semantic_paraphrase): What explains inconsistent accuracy loss in compressed language models? Analyzing noise introduction and layer-wise error accumulation. [ranks: BM25=None, Dense=4, Hybrid=6]
- `q0086` (keyword): multi-model EHR natural language query benchmark [ranks: BM25=None, Dense=1, Hybrid=2]
- `q0091` (keyword): code comment influence on large language model bug repair [ranks: BM25=None, Dense=1, Hybrid=2]

### `hybrid_success_dense_failure` (5)

- `q0051` (natural_question): Does temporal grounding precision fail to improve with model scale in streaming episodic memory? [ranks: BM25=1, Dense=None, Hybrid=1]
- `q0062` (keyword): RML mapping pruning for efficient SPARQL answering [ranks: BM25=1, Dense=None, Hybrid=1]
- `q0094` (keyword): spectral subgraph repair for planning graph stabilization [ranks: BM25=1, Dense=None, Hybrid=1]

### `dense_success_bm25_failure` (6)

- `q0073` (semantic_paraphrase): Using tabular metadata in model documentation to retrieve varied and comparable AI systems [ranks: BM25=None, Dense=10, Hybrid=None]
- `q0082` (semantic_paraphrase): What explains inconsistent accuracy loss in compressed language models? Analyzing noise introduction and layer-wise error accumulation. [ranks: BM25=None, Dense=4, Hybrid=6]
- `q0086` (keyword): multi-model EHR natural language query benchmark [ranks: BM25=None, Dense=1, Hybrid=2]

### `bm25_success_dense_failure` (5)

- `q0051` (natural_question): Does temporal grounding precision fail to improve with model scale in streaming episodic memory? [ranks: BM25=1, Dense=None, Hybrid=1]
- `q0062` (keyword): RML mapping pruning for efficient SPARQL answering [ranks: BM25=1, Dense=None, Hybrid=1]
- `q0094` (keyword): spectral subgraph repair for planning graph stabilization [ranks: BM25=1, Dense=None, Hybrid=1]

### `all_methods_failure` (4)

- `q0090` (semantic_paraphrase): unaddressed challenges and potential solutions for lowering environmental impact of large-scale satellite data analytics on distributed systems [ranks: BM25=None, Dense=None, Hybrid=None]
- `q0113` (semantic_paraphrase): Improving swarm optimization for routing tasks via predictive sensory minimization models [ranks: BM25=None, Dense=None, Hybrid=None]
- `q0121` (semantic_paraphrase): Improving clinical document search by combining biomedical knowledge graphs and generated training examples. [ranks: BM25=None, Dense=None, Hybrid=None]

## Limitations

Benchmark v1 is a synthetic known-item retrieval benchmark generated from source-paper titles and abstracts. It is not equivalent to human relevance judgments or a standard public IR benchmark. Latency is specific to the recorded machine and index configuration.
