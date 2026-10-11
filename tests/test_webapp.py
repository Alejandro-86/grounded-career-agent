from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from career_agent.llm.fakes import FaultInjectingLLM, HeuristicLLM
from career_agent.observability.tracing import NoopTracer
from career_agent.storage.store import MemoryRunStore
from career_agent.webapp.main import create_app

JD = "Role\nRequirements:\n- Kafka event streaming\n- Production Go services\nNice to have:\n- Snowflake"


@pytest.fixture()
def client(profile):
    app = create_app(
        profile=profile,
        llm_factory=lambda: HeuristicLLM(profile),
        store=MemoryRunStore(),
        tracer=NoopTracer(),
    )
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "Grounded Career Agent" in r.text


def test_analyze_then_fetch_everything(client):
    r = client.post("/analyze", json={"title": "t", "jd_text": JD})
    assert r.status_code == 200
    run = r.json()["result"]
    assert [m["verdict"] for m in run["matches"]] == ["match", "gap", "gap"]

    rid = run["run_id"]
    assert client.get(f"/runs/{rid}").json()["run_id"] == rid
    assert any(x["run_id"] == rid for x in client.get("/runs").json())
    assert "Fit report" in client.get(f"/runs/{rid}/report").text
    cv = client.get(f"/runs/{rid}/cv").text
    assert "Sam Okafor" in cv and "Kafka" in cv and "Go services" not in cv


def test_unknown_run_is_404(client):
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/cv").status_code == 404


def test_short_or_unparseable_jd_is_rejected(client):
    assert client.post("/analyze", json={"jd_text": "short"}).status_code == 422
    r = client.post("/analyze", json={"jd_text": "a long paragraph with no bullet points at all."})
    assert r.status_code == 422 and "no requirements" in r.json()["detail"]


def test_guardrail_toggle_is_visible_through_the_api(profile):
    faulty = lambda: FaultInjectingLLM(HeuristicLLM(profile), p_overclaim=1.0)  # noqa: E731
    c = TestClient(
        create_app(profile=profile, llm_factory=faulty, store=MemoryRunStore(), tracer=NoopTracer())
    )
    jd = "Role\nRequirements:\n- Production Go services\n- Azure deployments"
    on = c.post("/analyze", json={"jd_text": jd, "guardrail": True}).json()["result"]
    off = c.post("/analyze", json={"jd_text": jd, "guardrail": False}).json()["result"]
    assert {m["verdict"] for m in on["matches"]} == {"gap"}
    assert {m["verdict"] for m in off["matches"]} == {"match"}


def test_offline_eval_endpoint_persists(client):
    r = client.post("/evals/run", json={"fault_rate": 0.5})
    assert r.status_code == 200
    assert r.json()["configs"]["guarded"]["fabrication_rate"] == 0.0
    assert len(client.get("/evals").json()) == 1
    assert client.post("/evals/run", json={"fault_rate": 3}).status_code == 422
