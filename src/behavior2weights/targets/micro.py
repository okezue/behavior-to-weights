from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from behavior2weights.models.microtransformer import MicroTransformer,MicroTransformerConfig
from behavior2weights.schemas import TargetRecord
from behavior2weights.targets.base import TargetAdapter,resolvecheckpoint
class MicroTransformerAdapter(TargetAdapter):
    family="micro-transformer"
    def __init__(self,*,manifest_root:str|Path|None=None)->None:
        self.manifest_root=manifest_root
    def canload(self,record:TargetRecord)->bool:
        return record.family_id==self.family
    def load(self,record:TargetRecord,*,device:str|torch.device="cpu")->nn.Module:
        raw_config=record.metadata.get("model_config")
        if not isinstance(raw_config,dict):
            raise ValueError(f"Target {record.target_id} has no metadata.model_config")
        config=MicroTransformerConfig.fromdict(raw_config)
        path=resolvecheckpoint(record,manifest_root=self.manifest_root)
        return MicroTransformer.load(path,config,device=device)
