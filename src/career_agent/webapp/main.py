from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..agent.pipeline import FitAgent
from ..agent.render import render_cv_html, render_report_md
from ..config import build_llm, build_store
from ..domain.models import Profile, RunResult
from ..domain.profile import load_profile
from ..eval.harness import EvalReport, run_eval
from ..llm.base import LLMClient
from ..observability.tracing import Tracer, default_tracer
from ..storage.store import RunStore

STATIC = Path(__file__).parent / "static"


class AnalyzeRequest(BaseModel):
    title: str = Field(default="Untitled role", max_length=200)
    jd_text: str = Field(min_length=20, max_length=20_000)
    guardrail: bool = True


class EvalRequest(BaseModel):
    fault_rate: float = Field(default=0.25, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    result: RunResult
    report: str


def create_app(
    *,
    profile: Profile,
    llm_factory: Callable[[], LLMClient],
    store: RunStore,
    tracer: Tracer | None = None,
) -> FastAPI:
    app = FastAPI(title="grounded-career-agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "profile": profile.name, "model": llm_factory().model}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
        agent = FitAgent(llm_factory(), profile, guardrail=req.guardrail, tracer=tracer)
        try:
            result = agent.analyze(req.title, req.jd_text)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        store.save_run(result)
        return AnalyzeResponse(result=result, report=render_report_md(result))

    @app.get("/runs")
    def runs(limit: int = 50) -> list[dict]:
        return store.list_runs(limit=min(limit, 200))

    def _get(run_id: str) -> RunResult:
        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        return r

    @app.get("/runs/{run_id}", response_model=RunResult)
    def run(run_id: str) -> RunResult:
        return _get(run_id)

    @app.get("/runs/{run_id}/cv", response_class=HTMLResponse)
    def cv(run_id: str) -> str:
        return render_cv_html(profile, _get(run_id))

    @app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
    def report(run_id: str) -> str:
        return render_report_md(_get(run_id))

    @app.get("/evals", response_model=list[EvalReport])
    def evals() -> list[EvalReport]:
        return store.list_evals()

    @app.post("/evals/run", response_model=EvalReport)
    def run_offline_eval(req: EvalRequest) -> EvalReport:
        """Fault-injection eval against the offline model (the live eval is a CLI job)."""
        offline = lambda: build_llm("offline", profile, req.fault_rate)  # noqa: E731
        rep = run_eval(offline, profile, params={"mode": "offline", "fault_rate": req.fault_rate})
        store.save_eval(rep)
        return rep

    return app


def create_default_app() -> FastAPI:
    profile = load_profile()
    mode = os.getenv("CAREER_AGENT_LLM", "offline")
    return create_app(
        profile=profile,
        llm_factory=lambda: build_llm(mode, profile),
        store=build_store(),
        tracer=default_tracer(enabled=os.getenv("CAREER_AGENT_TRACING", "1") == "1"),
    )
