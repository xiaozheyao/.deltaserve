from collections import deque
from typing import Deque
import random
from vllm.sequence import SequenceGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class Policy:
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        occurences: dict = None,
    ) -> float:
        raise NotImplementedError

    def get_most_wanted(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ):
        # sort by fcfs, then get the first seq_groups
        seq_groups = sorted(
            seq_groups, key=lambda seq_group: seq_group.metrics.arrival_time
        )
        if len(seq_groups) == 0:
            return None
        return seq_groups[0]

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
        occurences: dict = None,
        available_deltas: list = None,
        most_wanted: SequenceGroup = None,
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(
                    now,
                    seq_group,
                    occurences=occurences,
                    available_deltas=available_deltas,
                    most_wanted=most_wanted,
                ),
                reverse=True,
            )
        )


class FCFS(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        occurences: dict = None,
        available_deltas: list = None,
        most_wanted=None,
    ) -> float:
        return now - seq_group.metrics.arrival_time


class DeltaServe(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        occurences: dict = None,
        available_deltas: list = None,
        most_wanted=None,
    ) -> float:
        if occurences is None:
            # fall back to fcfs
            return now - seq_group.metrics.arrival_time
        if available_deltas is None:
            return (
                occurences[seq_group.delta_int_id]
                + now
                - seq_group.metrics.arrival_time
            )
        if most_wanted is not None:
            most_wanted_bonus = (
                100 if seq_group.delta_int_id == most_wanted.delta_int_id else 0
            )
        else:
            most_wanted_bonus = 0
        # available_bonus = 10000 if seq_group.delta_int_id in available_deltas else 0
        logger.info("[scheduler]: most wanted bonus: ", most_wanted_bonus)
        logger.info(
            "[scheduler]: fcfs priority: ", now - seq_group.metrics.arrival_time
        )
        return now - seq_group.metrics.arrival_time + most_wanted_bonus


class RandomPolicy(Policy):
    def get_priority(self, now: float, seq_group: SequenceGroup) -> float:
        return random.random()


class PolicyFactory:
    _POLICY_REGISTRY = {"fcfs": FCFS, "random": RandomPolicy, "deltaserve": DeltaServe}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
