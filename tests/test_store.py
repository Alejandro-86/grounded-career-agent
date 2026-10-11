from __future__ import annotations

import pytest

from career_agent.agent.pipeline import FitAgent
from career_agent.eval.harness import EvalReport, jd_text, load_eval_set
from career_agent.llm.fakes import HeuristicLLM
from career_agent.storage.store import DeltaRunStore, MemoryRunStore


@pytest.fixture()
def result(profile):
    jd = load_eval_set()[0]
    return FitAgent(HeuristicLLM(profile), profile).analyze(
        jd["title"], jd_text(jd), run_id="run-a"
    )


def test_memory_store_roundtrip(result):
    s = MemoryRunStore()
    s.save_run(result)
    assert s.get_run("run-a") == result
    assert s.get_run("nope") is None
    assert s.list_runs()[0]["n_gap"] == result.counts["gap"]


def test_memory_store_evals():
    s = MemoryRunStore()
    s.save_eval(EvalReport(llm_model="m", n_jds=1, n_gold=1, configs={"guarded": {"x": 1.0}}))
    assert s.list_evals()[0].llm_model == "m"


@pytest.fixture(scope="module")
def spark():
    pyspark = pytest.importorskip("pyspark")
    assert pyspark
    from career_agent.storage.spark import get_local_spark

    s = get_local_spark("career-agent-tests")
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def test_delta_store_roundtrip_history_and_time_travel(spark, tmp_path, result):
    store = DeltaRunStore(spark, str(tmp_path))
    assert store.get_run("run-a") is None and store.list_runs() == []  # before any write

    store.save_run(result)
    store.save_run(result.model_copy(update={"run_id": "run-b"}))

    back = store.get_run("run-a")
    assert back is not None and back.model_dump() == result.model_dump()
    assert {r["run_id"] for r in store.list_runs()} == {"run-a", "run-b"}

    versions = store.history("runs")
    assert [v["version"] for v in versions] == [1, 0]
    assert all(v["operation"] == "WRITE" for v in versions)
    assert store.read_version("runs", 0).count() == 1  # time travel to before run-b
    assert store.read_version("matches", 1).count() == 2 * len(result.matches)


def test_delta_matches_table_records_guardrail_activity(spark, tmp_path, profile):
    from career_agent.llm.fakes import FaultInjectingLLM

    llm = FaultInjectingLLM(HeuristicLLM(profile), p_overclaim=1.0)
    jd = load_eval_set()[2]
    r = FitAgent(llm, profile).analyze("t", jd_text(jd), run_id="faulty")
    store = DeltaRunStore(spark, str(tmp_path))
    store.save_run(r)
    df = store.spark.read.format("delta").load(store.path("matches"))
    intervened = df.filter("size(violations) > 0").count()
    assert intervened == sum(bool(m.violations) for m in r.matches) > 0


def test_delta_eval_roundtrip(spark, tmp_path):
    store = DeltaRunStore(spark, str(tmp_path))
    rep = EvalReport(
        llm_model="m",
        n_jds=1,
        n_gold=2,
        configs={
            "baseline_no_guardrail": {"fabrication_rate": 0.4},
            "guarded": {"fabrication_rate": 0.0, "verdict_accuracy": 1.0},
        },
    )
    store.save_eval(rep)
    got = store.list_evals()
    assert got[0].eval_id == rep.eval_id
    assert got[0].configs["guarded"]["verdict_accuracy"] == 1.0
