from __future__ import annotations
from abc import ABC,abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
from torch import Tensor,nn
from behavior2weights.schemas import TargetRecord
class TargetAdapter(ABC):
    family:str
    @abstractmethod
    def canload(self,record:TargetRecord)->bool:
        raise NotImplementedError
    @abstractmethod
    def load(self,record:TargetRecord,*,device:str|torch.device="cpu")->nn.Module:
        raise NotImplementedError
    def state_dict(self,record:TargetRecord)->Mapping[str,Tensor]:
        return self.load(record,device="cpu").state_dict()
    def architecturemetadata(self,record:TargetRecord)->dict[str,Any]:
        return dict(record.metadata.get("model_config",{}))
class TargetRegistry:
    def __init__(self)->None:
        self._adapters:list[TargetAdapter]=[]
    def register(self,adapter:TargetAdapter)->None:
        self._adapters.append(adapter)
    def adapterfor(self,record:TargetRecord)->TargetAdapter:
        for adapter in self._adapters:
            if adapter.canload(record):
                return adapter
        raise LookupError(f"No target adapter registered for family_id={record.family_id!r}")
    def load(self,record:TargetRecord,*,device:str|torch.device="cpu")->nn.Module:
        return self.adapterfor(record).load(record,device=device)
def resolvecheckpoint(record:TargetRecord,*,manifest_root:str|Path|None=None)->Path:
    path=Path(record.checkpoint_path)
    if path.is_absolute()or manifest_root is None:
        return path
    return Path(manifest_root)/path
