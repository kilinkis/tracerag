# TraceRAG

TraceRAG is an open-source, evidence-first retrieval augmented generation system. It answers
questions over technical documentation while exposing its retrieved evidence, citations, quality
metrics, latency, and cost.

The core keeps retrieval mechanics explicit and measurable. Framework integrations, including
LangChain, remain replaceable adapters and are evaluated against the same benchmark.

## Current status

Milestone 0 is in progress. The repository currently provides:

- A typed FastAPI service with a health endpoint
- PostgreSQL 17 with the pgvector extension
- Reproducible Python dependencies through uv
- A smoke test and GitHub Actions workflow
- A milestone-based [delivery roadmap](docs/roadmap.md)

No retrieval or generation behavior has been implemented yet.

## Local development

Prerequisites:

- Docker with Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Install dependencies and run the tests:

```bash
uv sync
uv run pytest
```

Start the complete local stack:

```bash
cp .env.example .env
docker compose up --build
```

Then open:

- Health check: <http://localhost:8000/health>
- Interactive API documentation: <http://localhost:8000/docs>

Stop the services with `docker compose down`. The PostgreSQL data remains in the named Docker
volume between restarts.

## Initial architecture

```text
documents -> parser -> chunks -> embeddings -> PostgreSQL/pgvector
                                                |
question  -> query embedding -> retrieval -------+
                                                |
                         evidence -> generator -> cited answer
```

The components will communicate through small project-owned interfaces. Provider and framework
integrations—including LangChain—will remain replaceable adapters.

## Engineering principles

- Evaluate retrieval separately from answer generation.
- Prefer explicit evidence and abstention over plausible unsupported answers.
- Add complexity only when a benchmark demonstrates its value.
- Keep ingestion reproducible and the evaluation corpus version-controlled.
- Track quality, latency, and inference usage together.

## Documentation sources used by the scaffold

- [FastAPI first steps](https://fastapi.tiangolo.com/tutorial/first-steps/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Python `pyproject.toml` guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Pydantic settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [uv project guide](https://docs.astral.sh/uv/guides/projects/)
- [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/)
- [Docker Compose startup ordering](https://docs.docker.com/compose/how-tos/startup-order/)
- [pgvector](https://github.com/pgvector/pgvector)
