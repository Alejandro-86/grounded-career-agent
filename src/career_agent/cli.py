from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent.pipeline import FitAgent
from .agent.render import render_cv_html, render_report_md
from .config import build_llm, build_store
from .domain.profile import load_profile
from .eval.harness import format_report, jd_text, load_eval_set, run_eval
from .observability.tracing import MLflowTracer, default_tracer


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="offline: deterministic stand-in model; live: Claude (needs ANTHROPIC_API_KEY)",
    )
    p.add_argument(
        "--fault-rate",
        type=float,
        default=0.0,
        help="inject model faults (overclaims, bad citations, inflated numbers) at this rate",
    )
    p.add_argument("--store", choices=["memory", "delta"], default="memory")
    p.add_argument("--warehouse", default="./warehouse")
    p.add_argument("--trace", action="store_true", help="record MLflow traces to ./mlflow.db")


def cmd_analyze(a: argparse.Namespace) -> int:
    profile = load_profile()
    if a.jd:
        title, text = a.title or Path(a.jd).stem, Path(a.jd).read_text()
    else:
        samples = {j["id"]: j for j in load_eval_set()}
        if a.sample not in samples:
            print(f"unknown sample {a.sample!r}; choose from {sorted(samples)}", file=sys.stderr)
            return 2
        jd = samples[a.sample]
        title, text = f"{jd['title']} / {jd['company']}", jd_text(jd)
    agent = FitAgent(
        build_llm(a.mode, profile, a.fault_rate),
        profile,
        guardrail=not a.no_guardrail,
        tracer=default_tracer(a.trace),
    )
    result = agent.analyze(title, text)
    build_store(a.store, a.warehouse).save_run(result)
    print(render_report_md(result))
    print("\nCV bullets:")
    for b in result.bullets:
        print(f"  - {b.text}  [{', '.join(b.fact_ids)}]")
    for d in result.dropped_bullets:
        print(f"  x dropped ungrounded bullet ({'; '.join(d.violations)})")
    if a.cv:
        Path(a.cv).write_text(render_cv_html(profile, result))
        print(f"\nCV HTML written to {a.cv}")
    print(f"\nrun {result.run_id}" + (f" | trace {result.trace_id}" if result.trace_id else ""))
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    profile = load_profile()
    tracer = MLflowTracer() if a.trace else None
    report = run_eval(
        lambda: build_llm(a.mode, profile, a.fault_rate),
        profile,
        tracer=tracer,
        params={"mode": a.mode, "fault_rate": a.fault_rate},
    )
    build_store(a.store, a.warehouse).save_eval(report)
    if tracer is not None:
        for name, metrics in report.configs.items():
            tracer.log_eval(
                f"eval-{report.eval_id}-{name}",
                {"model": report.llm_model, **report.params},
                metrics,
            )
    print(format_report(report))
    if a.mode == "offline":
        print(
            "\nNote: offline mode measures guardrail containment of injected faults, "
            "not any model's judgement quality. Use --mode live for that."
        )
    return 0


def cmd_history(a: argparse.Namespace) -> int:
    store = build_store("delta", a.warehouse)
    for table in ("runs", "matches", "evals"):
        print(f"\n[{table}] Delta history")
        for h in store.history(table):  # type: ignore[attr-defined]
            print(f"  v{h['version']:<3} {h['timestamp']}  {h['operation']}")
    print("\nlatest runs:")
    for r in store.list_runs(10):
        print(
            f"  {r['run_id']}  {r['score']:.0%}  {r['title']}  (gaps {r['n_gap']}, violations {r['n_violations']})"
        )
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "career_agent.webapp.main:create_default_app", factory=True, host=a.host, port=a.port
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="career-agent", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    an = sub.add_parser("analyze", help="fit/gap report and tailored CV for one JD")
    g = an.add_mutually_exclusive_group()
    g.add_argument("--jd", help="path to a JD text file (bulleted requirements)")
    g.add_argument("--sample", default="northwind", help="built-in fictional JD id")
    an.add_argument("--title")
    an.add_argument(
        "--no-guardrail", action="store_true", help="accept raw model output (baseline)"
    )
    an.add_argument("--cv", help="write the tailored CV HTML to this path")
    _common(an)
    an.set_defaults(fn=cmd_analyze)

    ev = sub.add_parser("eval", help="run the labelled eval, guardrail off vs on")
    _common(ev)
    ev.set_defaults(fn=cmd_eval)

    hi = sub.add_parser("history", help="show Delta table history and recent runs")
    hi.add_argument("--warehouse", default="./warehouse")
    hi.set_defaults(fn=cmd_history)

    sv = sub.add_parser("serve", help="run the FastAPI service")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
