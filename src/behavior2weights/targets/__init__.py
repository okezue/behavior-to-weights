from behavior2weights.targets.base import TargetAdapter, TargetRegistry
from behavior2weights.targets.huggingface import HuggingFaceCausalLMAdapter
from behavior2weights.targets.micro import MicroTransformerAdapter

__all__ = [
    "HuggingFaceCausalLMAdapter",
    "MicroTransformerAdapter",
    "TargetAdapter",
    "TargetRegistry",
]
