"""Local, Delta-enabled SparkSession.

On a Databricks workspace none of this is needed: ``spark`` already exists with
Delta and Unity Catalog configured. The store below takes a SparkSession and a
base path, so pointing it at ``dbfs:/`` or a Unity Catalog volume is a
constructor argument, not a code change.
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession

DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.2.1"


def get_local_spark(app_name: str = "grounded-career-agent") -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    return (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.jars.packages", DELTA_PACKAGE)
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
