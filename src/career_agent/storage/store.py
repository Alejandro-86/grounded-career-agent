"""Persistence for runs and eval reports.

``DeltaRunStore`` writes three Delta tables (path-based, so it works the same on
a laptop and a Databricks volume):

  runs     one row per analysis, typed summary columns + the full JSON payload
  matches  one row per requirement judgement (queryable audit of guardrail activity)
  evals    one row per eval run, with the headline metrics as columns

Appends only. Delta's transaction log gives history and time travel for free.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from ..domain.models import RunResult
from ..eval.harness import EvalReport


class RunStore(Protocol):
    def save_run(self, result: RunResult) -> None: ...
    def get_run(self, run_id: str) -> RunResult | None: ...
    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]: ...
    def save_eval(self, report: EvalReport) -> None: ...
    def list_evals(self, limit: int = 20) -> list[EvalReport]: ...


def run_summary(r: RunResult) -> dict[str, Any]:
    c = r.counts
    return {
        "run_id": r.run_id,
        "created_at": r.created_at,
        "title": r.title,
        "llm_model": r.llm_model,
        "guardrail": r.guardrail_enabled,
        "score": r.score,
        "n_requirements": len(r.requirements),
        "n_match": c["match"],
        "n_partial": c["partial"],
        "n_gap": c["gap"],
        "n_violations": r.n_violations,
        "trace_id": r.trace_id,
    }


class MemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunResult] = {}
        self._evals: list[EvalReport] = []

    def save_run(self, result: RunResult) -> None:
        self._runs[result.run_id] = result

    def get_run(self, run_id: str) -> RunResult | None:
        return self._runs.get(run_id)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        return [run_summary(r) for r in rows[:limit]]

    def save_eval(self, report: EvalReport) -> None:
        self._evals.append(report)

    def list_evals(self, limit: int = 20) -> list[EvalReport]:
        return sorted(self._evals, key=lambda e: e.created_at, reverse=True)[:limit]


class DeltaRunStore:
    def __init__(self, spark: Any, base_path: str = "./warehouse"):
        from pyspark.sql import types as T

        self.spark = spark
        self.base = base_path.rstrip("/")
        self._T = T
        self.runs_schema = T.StructType(
            [
                T.StructField("run_id", T.StringType(), False),
                T.StructField("created_at", T.TimestampType(), False),
                T.StructField("title", T.StringType(), False),
                T.StructField("llm_model", T.StringType(), False),
                T.StructField("guardrail", T.BooleanType(), False),
                T.StructField("score", T.DoubleType(), False),
                T.StructField("n_requirements", T.IntegerType(), False),
                T.StructField("n_match", T.IntegerType(), False),
                T.StructField("n_partial", T.IntegerType(), False),
                T.StructField("n_gap", T.IntegerType(), False),
                T.StructField("n_violations", T.IntegerType(), False),
                T.StructField("trace_id", T.StringType(), True),
                T.StructField("payload", T.StringType(), False),
            ]
        )
        self.matches_schema = T.StructType(
            [
                T.StructField("run_id", T.StringType(), False),
                T.StructField("requirement_id", T.StringType(), False),
                T.StructField("requirement", T.StringType(), False),
                T.StructField("must_have", T.BooleanType(), False),
                T.StructField("verdict", T.StringType(), False),
                T.StructField("claimed_verdict", T.StringType(), False),
                T.StructField("fact_ids", T.ArrayType(T.StringType()), False),
                T.StructField("violations", T.ArrayType(T.StringType()), False),
                T.StructField("attempts", T.IntegerType(), False),
                T.StructField("downgraded", T.BooleanType(), False),
            ]
        )
        self.evals_schema = T.StructType(
            [
                T.StructField("eval_id", T.StringType(), False),
                T.StructField("created_at", T.TimestampType(), False),
                T.StructField("llm_model", T.StringType(), False),
                T.StructField("fabrication_baseline", T.DoubleType(), False),
                T.StructField("fabrication_guarded", T.DoubleType(), False),
                T.StructField("accuracy_guarded", T.DoubleType(), False),
                T.StructField("payload", T.StringType(), False),
            ]
        )

    def path(self, table: str) -> str:
        return f"{self.base}/{table}"

    def _append(self, rows: list[tuple[Any, ...]], schema: Any, table: str) -> None:
        self.spark.createDataFrame(rows, schema).write.format("delta").mode("append").save(
            self.path(table)
        )

    def _read(self, table: str) -> Any | None:
        from delta.tables import DeltaTable

        if not DeltaTable.isDeltaTable(self.spark, self.path(table)):
            return None
        return self.spark.read.format("delta").load(self.path(table))

    # ---- runs

    def save_run(self, result: RunResult) -> None:
        s = run_summary(result)
        self._append(
            [
                (
                    s["run_id"],
                    _naive(s["created_at"]),
                    s["title"],
                    s["llm_model"],
                    s["guardrail"],
                    float(s["score"]),
                    s["n_requirements"],
                    s["n_match"],
                    s["n_partial"],
                    s["n_gap"],
                    s["n_violations"],
                    s["trace_id"],
                    result.model_dump_json(),
                )
            ],
            self.runs_schema,
            "runs",
        )
        by_id = {r.id: r for r in result.requirements}
        self._append(
            [
                (
                    result.run_id,
                    m.requirement_id,
                    by_id[m.requirement_id].text,
                    by_id[m.requirement_id].must_have,
                    m.verdict.value,
                    m.claimed_verdict.value,
                    list(m.fact_ids),
                    list(m.violations),
                    m.attempts,
                    m.downgraded,
                )
                for m in result.matches
            ],
            self.matches_schema,
            "matches",
        )

    def get_run(self, run_id: str) -> RunResult | None:
        df = self._read("runs")
        if df is None:
            return None
        rows = df.filter(df.run_id == run_id).select("payload").limit(1).collect()
        return RunResult.model_validate_json(rows[0].payload) if rows else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        df = self._read("runs")
        if df is None:
            return []
        rows = df.drop("payload").orderBy(df.created_at.desc()).limit(limit).collect()
        return [r.asDict() for r in rows]

    # ---- evals

    def save_eval(self, report: EvalReport) -> None:
        base = report.configs.get("baseline_no_guardrail", {})
        guarded = report.configs.get("guarded", {})
        self._append(
            [
                (
                    report.eval_id,
                    _naive(report.created_at),
                    report.llm_model,
                    float(base.get("fabrication_rate", 0.0)),
                    float(guarded.get("fabrication_rate", 0.0)),
                    float(guarded.get("verdict_accuracy", 0.0)),
                    report.model_dump_json(),
                )
            ],
            self.evals_schema,
            "evals",
        )

    def list_evals(self, limit: int = 20) -> list[EvalReport]:
        df = self._read("evals")
        if df is None:
            return []
        rows = df.orderBy(df.created_at.desc()).limit(limit).select("payload").collect()
        return [EvalReport.model_validate_json(r.payload) for r in rows]

    # ---- Delta-specific

    def history(self, table: str = "runs") -> list[dict[str, Any]]:
        from delta.tables import DeltaTable

        if not DeltaTable.isDeltaTable(self.spark, self.path(table)):
            return []
        h = DeltaTable.forPath(self.spark, self.path(table)).history()
        return [r.asDict() for r in h.select("version", "timestamp", "operation").collect()]

    def read_version(self, table: str, version: int) -> Any:
        return self.spark.read.format("delta").option("versionAsOf", version).load(self.path(table))


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo else dt
