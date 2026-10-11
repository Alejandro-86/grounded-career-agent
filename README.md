# grounded-career-agent

An agent that reads a job description, tells a candidate honestly where they fit and where they don't, and writes a tailored CV, **without ever claiming something the candidate's profile can't back up.**

The interesting part isn't the LLM call. It's everything around it: a code-enforced grounding guardrail the model cannot argue with, an eval harness that measures how often an unguarded model fabricates, Delta Lake persistence of every judgement, and MLflow traces of every step.

```
JD text
  │
  ▼
extract_requirements ──► skills = model's list ∪ deterministic taxonomy scan
  │                      (a model can't shrink a requirement to make it easier to cover)
  ▼
for each requirement:
  retrieve candidate facts (skill overlap first, wording second)
  │
  ▼
  judge_match (LLM proposes) ──► GUARDRAIL G1–G4 ──► ok?  ──► accept
                                        │                └─ no ──► feed violations back,
                                        │                          retry (bounded) ──► else
                                        │                          deterministic downgrade
  ▼
for each supported requirement:
  write_bullet (LLM proposes) ──► GUARDRAIL B1–B4 ──► ok? ──► accept │ retry │ drop (never "repaired")
  ▼
RunResult ──► Delta tables (runs, matches, evals) + MLflow trace + FastAPI + HTML CV
```

## Why a guardrail and not a better prompt

Asked to tailor a CV to a JD, an LLM will happily round up: a personal project becomes "production experience", a missing skill gets glossed over, a number grows. A prompt can ask it not to. It can't *guarantee* it. So the model only **proposes**; plain Python decides what is **allowed**:

| Rule | Enforces |
|---|---|
| G1 | every cited fact id exists in the profile |
| G2 | a match/partial verdict cites at least one fact |
| G3 | **coverage ceiling**: MATCH needs the cited facts to cover *every* skill the requirement names; some skills caps at PARTIAL; none caps at GAP |
| G4 | **evidence ceiling**: only *production* evidence can support MATCH. Personal projects and adjacent exposure cap at PARTIAL |
| B1 | a CV bullet cites at least one real fact |
| B2 | every number in a bullet appears in the cited facts |
| B3 | every skill a bullet mentions belongs to a cited fact |
| B4 | a bullet never mentions a skill the profile declares as a gap |

When a proposal is rejected, the violations go back to the model for a bounded self-correction (default 2 retries). If it still fails, the run **degrades to the honest answer**: a lower verdict, or a dropped bullet. It never ships the unsupported claim and never silently edits it. Declared gaps are structural: loading a profile that tags a skill it also declares as a gap is an error, so a gap can't be covered by accident.

## Quickstart

Offline mode needs no API key. It uses a deterministic stand-in model so the whole pipeline, eval and API run anywhere.

```bash
make setup                                   # uv venv (Python 3.11) + install
make test                                    # 58 tests
career-agent analyze --sample helix --fault-rate 0.5
career-agent eval --fault-rate 0.25
career-agent serve                           # http://127.0.0.1:8000  (UI at /, docs at /docs)
```

Live mode uses Claude via the Anthropic API (structured output through forced tool use):

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY
career-agent analyze --mode live --jd my_jd.txt --cv tailored.html
career-agent eval --mode live --store delta --trace
```

A JD file is plain text with bulleted requirements. Headers containing "nice", "bonus" or "preferred" mark the bullets below them optional.

## Evaluation

`career-agent eval` runs 6 fictional JDs (56 hand-labelled requirements) twice: guardrail **off** (raw model behaviour) and **on**. Gold labels follow the evidence ladder, and a test re-derives every label straight from the profile, independently of the agent code.

The headline metric is the **fabrication rate**: of the requirements whose correct verdict is GAP, how many did the agent claim some coverage for.

Offline model with seeded faults injected at the ways real models fail (over-claiming a verdict, citing a fact that doesn't exist, inflating a number, slipping in a skill the candidate lacks):

| metric (lower = better where marked) | fault rate 0.25: no guardrail | guarded | fault rate 1.0: no guardrail | guarded |
|---|---|---|---|---|
| verdict accuracy | 0.911 | 1.000 | 0.518 | 1.000 |
| **fabrication** (↓) | 0.231 | **0.000** | 1.000 | **0.000** |
| overclaim rate (↓) | 0.089 | 0.000 | 0.482 | 0.000 |
| unsupported citations (↓) | 0.161 | 0.000 | 0.464 | 0.000 |
| ungrounded CV bullets (↓) | 0.275 | 0.000 | 1.000 | 0.000 |
| guardrail intervention rate | 0.000 | 0.250 | 0.000 | 0.732 |
| downgraded after retries | 0.000 | 0.000 | 0.000 | 0.536 |
| mean attempts per judgement | 1.000 | 1.302 | 1.000 | 2.335 |

Each row is reproducible: `career-agent eval --fault-rate 0.25` and `--fault-rate 1.0`.

### What this does and doesn't show

- **It shows** the guardrail contains every fault class, that bounded retries recover a good share of them (at 0.25 nothing needed a downgrade; at 1.0 about half did), and that on a clean model the guardrail costs nothing (underclaim rate 0).
- **It does not show** how good any real model's judgement is. The offline model uses the same coverage logic as the guardrail, so 1.000 guarded accuracy is partly by construction. Live-model quality is what `eval --mode live` measures, and **that hasn't been run yet**; it needs an API key. Treat the table as a stress test of the guardrail, not a benchmark of Claude.
- Bullet grounding in the "no guardrail" column is audited with the guardrail's own checker, so it measures how often raw output would have been rejected.

## Delta Lake and MLflow

Every run is appended to three Delta tables (path-based, so a laptop directory and a Databricks volume behave the same):

| table | grain | use |
|---|---|---|
| `runs` | one row per analysis | typed summary columns plus the full JSON payload |
| `matches` | one row per requirement judgement | query guardrail activity: `size(violations) > 0`, `downgraded`, `claimed_verdict` vs `verdict` |
| `evals` | one row per eval run | headline metrics as columns, full report as JSON |

```bash
career-agent analyze --sample brightwave --fault-rate 0.25 --store delta
career-agent history          # Delta transaction-log versions per table + recent runs
```

`DeltaRunStore` takes a `SparkSession` and a base path, so pointing it at a workspace is a constructor argument. It has **not** been run on a real Databricks workspace; locally it runs on PySpark 3.5 with `delta-spark` 3.2.

With `--trace`, each analysis is recorded as an MLflow trace (root span `analyze`, then `extract_requirements`, one `judge_match:rNN` and `write_bullet:rNN` per step, with verdicts and violations as span outputs), and each eval logs its metrics and parameters as MLflow runs. Tracing is behind a small interface and degrades to a no-op if the backend is unavailable, so it can't take a run down.

## API

| endpoint | |
|---|---|
| `POST /analyze` | `{title, jd_text, guardrail}` returns the full result and a fit report |
| `GET /runs`, `/runs/{id}` | history and a single run |
| `GET /runs/{id}/cv`, `/runs/{id}/report` | tailored CV (HTML) and fit report (markdown) |
| `POST /evals/run`, `GET /evals` | offline fault-injection eval; stored eval reports |
| `GET /health` | status and active model |

`guardrail: false` returns the raw model output, which is what makes the failure mode demonstrable. The UI at `/` is a single static page.

## Layout

```
src/career_agent/
  domain/        models, taxonomy (deterministic skill scan), guardrail (G1–G4, B1–B4), profile loading
  agent/         pipeline (explicit typed steps + bounded retry), retrieval, prompts, CV/report rendering
  llm/           LLMClient protocol, Anthropic client (forced tool use), fault-injecting test doubles
  eval/          harness: alignment, metrics, baseline-vs-guarded report
  storage/       Delta store + in-memory store, local Spark session
  observability/ tracing interface, MLflow implementation
  webapp/        FastAPI service + static UI
data/            fictional profile, skills taxonomy, labelled eval set
tests/           58 tests
```

## Design notes and limitations

- **Explicit steps, not a free tool loop.** The task has a fixed shape (extract, judge each, write each), so a typed pipeline is easier to trace, test and bound than an open-ended agent loop. Each step has one job and one schema.
- **Skills are checked by code, not trusted.** The skill set for a requirement is the union of the model's list and a regex scan against the taxonomy. A model that "forgets" the hard skill can't make a requirement easier.
- **The evidence ladder is deliberately strict.** A fully on-topic personal project still caps at PARTIAL; only production work earns MATCH. That's a product decision, and it makes honest scores lower than a generous reader might give.
- **Alternatives in a requirement are treated as all-required.** "Delta Lake or Iceberg" needs both covered to MATCH, which errs conservative. The eval set avoids "or" requirements; handling them properly is future work.
- **Coverage is skill-level.** The taxonomy can't tell Spark batch from Spark Structured Streaming, so a requirement that hinges on that distinction would over-credit.
- **Retrieval is lexical** (skill overlap plus token overlap, top 5). Enough for a 19-fact profile; a larger one would want embeddings, and the guardrail would stay unchanged.
- **The profile is fictional** (`data/profile/sam_okafor.json`). Replace it with your own; loading checks that a fact's text never mentions a skill it doesn't tag, and that no declared gap is also claimed.
