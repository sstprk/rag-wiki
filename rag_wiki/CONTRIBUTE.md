# Contributing to rag-wiki

Thank you for your interest in contributing. This project explores a hybrid retrieval architecture that combines RAG systems with user-driven knowledge curation. Contributions are expected to maintain clarity, modularity, and architectural integrity.

---

## Contribution Principles

* Keep components **decoupled** (storage, lifecycle, retriever, adapters).
* Follow the **StateStore abstraction strictly** — no direct dependency on concrete backends.
* Prefer **explicit logic over implicit behavior**.
* Ensure **test coverage for every new feature or change**.
* Avoid introducing unnecessary dependencies.

---

## Development Setup

```bash
git clone https://github.com/sstprk/rag-wiki.git
cd rag-wiki

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -e .
pip install pytest
```

Run tests:

```bash
pytest tests/ -v
```

All tests must pass before submitting a contribution.

---

## Project Structure Overview

```
rag_wiki/
├── storage/        # StateStore implementations (SQLite, Memory, etc.)
├── lifecycle/      # State machine, decay engine, fetch counter
├── retriever.py    # HybridRetriever (core orchestration)
├── transparency/   # Provenance and context tracing
├── adapters/       # Framework integrations (LangChain, etc.)
```

---

## How to Contribute

### 1. Fork & Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

* Follow existing code patterns and naming conventions
* Add type hints and docstrings
* Keep files under ~250 lines when possible

### 3. Add Tests

* Every new module must include tests
* Use in-memory or mock backends only
* Avoid external services in tests

### 4. Run Full Test Suite

```bash
pytest tests/ -v
```

### 5. Submit Pull Request

Include:

* Clear description of the change
* Why it’s needed
* Any tradeoffs or limitations

---

## Coding Guidelines

* Python ≥ 3.10
* Use `datetime.now(UTC)` instead of deprecated methods
* No global state
* No hardcoded user IDs or document IDs
* Avoid circular imports
* Keep logic modular and composable

---

## Areas for Contribution

* Additional `StateStore` backends (Postgres, Redis, etc.)
* New framework adapters (LlamaIndex, Haystack)
* Retrieval optimization strategies
* UI/UX for provenance and transparency
* Performance improvements in lifecycle/decay logic

---

## Reporting Issues

When opening an issue, include:

* Description of the problem
* Steps to reproduce
* Expected vs actual behavior
* Environment details (Python version, OS, etc.)

---

## Notes

This project prioritizes **predictability over cleverness**. If a solution is difficult to reason about, it likely does not belong here.

---

## License

By contributing, you agree that your contributions will be licensed under the same license as this project.
