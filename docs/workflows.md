# Build, write, analyze, review, translate

Beyond research and batch tasks, the swarm handles the jobs people most often give an AI
assistant. Each one splits the work into parts that can be done in parallel, gives the parts
one shared plan so they fit together, and checks the result with something other than the
model's opinion.

| Command | What you get | How it is checked |
|---|---|---|
| `swarm build` | a new app (frontend, backend, CLI, library) in a folder | its own tests are installed and run; failures go to a fixer |
| `swarm write` | a long document (guide, report, proposal, docs) | section lengths, an editor pass that must keep every section |
| `swarm analyze` | a data report with charts from your CSV | every number must come from code that ran on your data |
| `swarm review` | a ranked list of bugs and security problems | every finding must quote code that is really in the file |
| `swarm doc` | a long document translated, summarized or rewritten | one glossary for all chunks; chunk length sanity checks |

All of them work with Claude or any OpenAI-compatible provider (the same `--model`,
`--base-url`, `--api-key-env` and `--price` flags as `swarm research`, or your `swarm.toml`),
and every run has a hard dollar cap (`--max-usd`).

## Build an app

```bash
swarm build "a todo app: React + Vite frontend, FastAPI backend with SQLite, tests for both" \
    --out todo --allow-execution
```

1. **Plan:** a lead engineer writes the spec: every file with its purpose, its *interface* (the
   functions, endpoints, JSON shapes and props other files rely on) and its dependencies, plus
   the setup and test commands.
2. **Write:** engineers write the files in parallel, in dependency order. Each sees the whole
   spec and the finished files it depends on, and codes to the contract. Python files must
   compile and JSON files must parse, or the file is rewritten.
3. **Run:** with `--allow-execution`, the setup commands (`pip install`, `npm install`) and the
   tests run for real. Python projects get their own `.venv`.
4. **Fix:** if anything fails, one fixer reads the error output and the files involved, edits
   them, and the run repeats, up to `--fix-rounds` (default 3).

The result is an ordinary project folder: a README with how to run it, its tests, and
`.swarm/build.json` (the spec, every command and its output, and the cost). Without
`--allow-execution` the files are written but never run.

**What the build commands can do.** They come from the model, so each is checked before it
runs:
- **No shell:** `&&`, pipes, redirects and `$VARS` are refused, not interpreted.
- **Allowlisted tools only:** `npm`, `pip`, `pytest`, `node`, `go`, `cargo` and the like.
  `sudo` and absolute program paths are refused.
- **Inside the project only:** commands start in the project folder, never outside it.
- **No credentials:** the environment has no API keys, and `HOME` is a private folder inside
  the project, so `~/.npmrc` and similar credential files aren't used.
- **Time limits:** a command that runs too long is killed, along with everything it started.

Installs still download packages from the internet, and the code can read files your user
can read. For a stronger boundary, run inside Docker.

**What to expect.** Small and medium apps (a CRUD API, a single-page frontend, a CLI tool, a
library with tests) usually come out working within the fix rounds. Big apps with many
screens or external services need iterations: build a core, then extend it. Multi-agent
coding is reliable here because of the shared spec and the real tests, not because the agents
talk to each other (they don't).

## Write a document

```bash
swarm write "A beginner's guide to home solar: costs, installation, savings, pitfalls" \
    --words 3000 --out solar-guide.md
swarm write "Q3 board update" --context notes.md metrics.csv --out update.md
swarm research "heat pumps vs boilers" --out hp.json && \
    swarm write "A customer-facing explainer" --context hp.json --out explainer.md
```

An editor plans the outline, style guide and word targets. Writers draft every section in
parallel, each seeing the whole plan so sections don't overlap. A final editing pass unifies
the voice for documents up to about 8,000 words. Writers use only the brief and your
`--context` files and are told never to invent statistics or citations. For sourced writing,
run `swarm research` first and pass the report as context.

## Analyze data

```bash
swarm analyze sales.csv "What drives revenue, and what changed this year?" \
    --allow-code-execution --out report/report.md
```

The file is profiled in code: columns, types, ranges and sample rows. A lead analyst picks up
to `--questions` independent analyses. Analysts write Python for each one, and the code
**runs** in the sandbox on your data. Errors go back to the analyst for a fix, up to 3 tries.

The report is written only from the results the code printed. Each number in it is then
checked against those results; any number that isn't there is listed in a "Number check" note.
Charts, every analysis script, and the results the code printed (`results.json`) are saved in
`report/analysis/`. The number check accepts figures one simple step from computed ones (a
difference, a sum, a share or a percent change), but only when the arithmetic is right at the
precision shown. pandas and matplotlib are
used when your Python has them.

## Review code

```bash
swarm review ./my-repo --out review.md
swarm review src/ --focus "security" --out security.md
```

Every file (git-tracked, or found by a walk that skips `node_modules`, `.venv`, build output
and the like) goes to its own reviewer. Long files are split into overlapping windows. Each
finding must quote the code it's about; a finding whose quote isn't in the file is dropped,
and line numbers are corrected to where the quote actually is.

A second, skeptical pass then checks each finding against the file and the project files it
imports (two levels deep). A first reviewer who saw one file may have guessed wrong about a
module it uses. Findings that the related code disproves move to a "Refuted on a second check"
list at the end of the report, with the verifier's reason, so you can disagree. Findings that
depend on code nobody can see are marked "unconfirmed". `--no-verify` skips this pass. You get Markdown ranked by
severity, plus a `.json` file with the findings.

## Translate, summarize or rewrite long documents

```bash
swarm doc translate manual.md --to German --out manual.de.md
swarm doc summarize annual-report.pdf --words 500 --out summary.md
swarm doc rewrite terms.md --style "plain English a customer understands" --out terms-plain.md
```

The document is split at headings and paragraph breaks, never inside a code block. For
translating and rewriting, a glossary of key terms is fixed first, so every chunk renders
names the same way. Each chunk also gets the end of the previous one for context.
Summaries are made per chunk and then merged, in rounds for very long documents. PDFs need
`pip install pypdf`.
