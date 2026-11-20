# Contributing

Thanks for helping. Bug reports, docs fixes, new search providers and evaluation questions
are all welcome.

## Setup

```bash
git clone https://github.com/Chetan0246/multi-agent-swarm && cd multi-agent-swarm
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
make lint test
```

Everything in the test suite runs offline: Claude and TinyFish are replaced by fake HTTP
transports, so tests need no keys and cost nothing. Please keep it that way. If a change
can't be tested offline, explain why in the PR.

## Making a change

1. Open an issue first for anything larger than a small fix, so we can agree on the approach.
2. Keep pull requests focused, and add or update tests.
3. Run `make lint test`. CI runs ruff, mypy and pytest on Python 3.10–3.13.
4. Add a line to `CHANGELOG.md` under an "Unreleased" heading.

## Good first contributions

- **Search providers:** implement `SearchProvider` in `swarm/search/` (see `tinyfish.py`).
- **Evaluation questions:** add known-answer questions to `evals/questions.jsonl`, with
  stable answers and several accepted spellings.
- **Injection patterns:** extend `swarm/safety.py`, with a test case for each pattern.

## Paid API calls

Never add tests or CI steps that call paid APIs. Live checks belong behind an explicit flag
(like `swarm doctor --online`) and are run by people, not by CI.

By contributing you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
