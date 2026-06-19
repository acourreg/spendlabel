"""Base PyFlink job — shared streaming topology for every consumer.

Mirrors patternalarm's ``StreamProcessorJob.scala``:
  - Creates a ``StreamExecutionEnvironment``
  - Enables checkpointing (default 60 s, matching patternalarm)
  - Builds a Kafka source (Confluent Cloud SASL_SSL)
  - Subclasses override ``process_stream()`` to attach their classifier

TODO:
    - Implement ``create_kafka_source`` using PyFlink's Kafka connector
      with SASL_SSL properties from ``config.settings``.
    - Implement ``create_kafka_sink`` for writing classified results.
    - Wire watermark strategy (bounded out-of-orderness, 30 s lateness).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pyflink.datastream import StreamExecutionEnvironment


class BaseFlinkJob(ABC):
    """Abstract base for all CPV classification PyFlink jobs.

    Subclasses must implement :meth:`process_stream` to attach their
    specific classification logic to the datastream pipeline.
    """

    def __init__(self, job_name: str, source_topic: str, sink_topic: str, group_id: str) -> None:
        self.job_name = job_name
        self.source_topic = source_topic
        self.sink_topic = sink_topic
        self.group_id = group_id

    def run(self) -> None:
        """Build and execute the Flink pipeline."""
        env = StreamExecutionEnvironment.get_execution_environment()

        # TODO: enable checkpointing (mirror patternalarm 60 s interval)
        # env.enable_checkpointing(settings.flink.checkpointing_interval_ms)

        # TODO: create Kafka source from self.source_topic
        # TODO: call self.process_stream(source_stream)
        # TODO: sink classified results to self.sink_topic
        # TODO: env.execute(self.job_name)

        raise NotImplementedError("Base Flink job pipeline not yet wired")

    @abstractmethod
    def process_stream(self, stream: object) -> object:
        """Attach classification logic to the incoming datastream.

        Parameters
        ----------
        stream:
            PyFlink DataStream of raw CPV contract records.

        Returns
        -------
        DataStream of classified records (with predicted CPV category).
        """
        ...
