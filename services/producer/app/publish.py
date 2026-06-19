"""Kafka producer — reads CSV files from data/raw/ and publishes each row
to the ``cpv-raw`` topic on Confluent Cloud.

Each message is keyed by contract notice ID; the value is the full row as JSON.

Usage::

    cd services/producer/app && python publish.py

TODO:
    - Load CSV files from ``data/raw/`` with pandas.
    - Connect to Confluent Cloud Kafka using SASL_SSL credentials from config.
    - Serialise each row as JSON and publish to ``cpv-raw``.
    - Log publish rate (rows/sec) to stdout.
"""

from __future__ import annotations


def main() -> None:
    """Entry-point: read CSVs and publish rows to Kafka."""
    # TODO: implement CSV loading and Kafka publishing
    raise NotImplementedError("Producer not yet implemented")


if __name__ == "__main__":
    main()
