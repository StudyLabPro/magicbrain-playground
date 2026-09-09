"""Сценарии плейграунда."""

from . import act_compare, neurogenesis, self_repair, train

REGISTRY = {
    "train": train.run,
    "neurogenesis": neurogenesis.run,
    "self-repair": self_repair.run,
    "act": act_compare.run,
}

__all__ = ["REGISTRY", "train", "neurogenesis", "self_repair", "act_compare"]
