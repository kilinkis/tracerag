# TraceRAG delivery roadmap

Every milestone should leave the repository runnable and should add measurements before adding
complexity.

## Milestone 0 — Foundation

- [x] Python project and dependency management
- [x] FastAPI service with a health endpoint
- [x] Local PostgreSQL with pgvector
- [x] Test and CI foundations

## Milestone 1 — Naive RAG baseline

- [x] Define document and chunk domain models
- [x] Ingest a small, versioned Markdown corpus
- [x] Implement deterministic structural chunking
- [ ] Generate and persist embeddings
- [ ] Retrieve top-k chunks using cosine similarity
- [ ] Generate answers constrained to retrieved evidence
- [ ] Return source citations and abstain when evidence is missing

## Milestone 2 — Evaluation before optimization

- [ ] Curate 30–50 answerable and unanswerable questions
- [ ] Measure recall@k, reciprocal rank, citation accuracy, and abstention accuracy
- [ ] Record latency and inference usage
- [ ] Publish baseline results and representative failures

## Milestone 3 — Better retrieval

- [ ] Add PostgreSQL full-text search
- [ ] Fuse dense and lexical rankings with Reciprocal Rank Fusion
- [ ] Add metadata filters and reranking
- [ ] Benchmark each change against the baseline

## Milestone 4 — Framework comparison

- [ ] Implement a LangChain adapter behind the existing interfaces
- [ ] Run the same benchmark against custom and LangChain pipelines
- [ ] Document quality, complexity, latency, and maintainability trade-offs

## Milestone 5 — Public demo

- [ ] Create the public GitHub repository
- [ ] Build the evidence and retrieval-trace UI
- [ ] Add rate limits, caching, and a global inference budget
- [ ] Deploy within free-tier quotas
- [ ] Publish architecture, evaluation results, screenshots, and a short demo
