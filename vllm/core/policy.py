from collections import deque
from typing import Deque
import random
from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        occurences: dict=None,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
        occurences: dict=None,
        available_deltas: list=None,
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(
                    now, 
                    seq_group, 
                    occurences=occurences, 
                    available_deltas=available_deltas
                ),
                reverse=True,
            )
        )

class FCFS(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        occurences: dict=None,
        available_deltas: list=None,
    ) -> float:
        return now - seq_group.metrics.arrival_time

class PopularFirst(Policy):
    
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        occurences: dict=None,
    ) -> float:
        return now - seq_group.metrics.arrival_time

class DeltaServe(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        occurences: dict=None,
        available_deltas: list=None,
    ) -> float:
        if occurences is None:
            return now - seq_group.metrics.arrival_time
        if available_deltas is None:
            return occurences[seq_group.delta_int_id] + now - seq_group.metrics.arrival_time
        available_bonus = 10 if seq_group.delta_int_id in available_deltas else 0
        return available_bonus + occurences[seq_group.delta_int_id] + now - seq_group.metrics.arrival_time
        
class RandomPolicy(Policy):
    def get_priority(self, now: float, seq_group: SequenceGroup) -> float:
        return random.random()

class PolicyFactory:
    _POLICY_REGISTRY = {
        "fcfs": FCFS,
        "popularity": PopularFirst,
        "random": RandomPolicy,
        "deltaserve": DeltaServe
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
