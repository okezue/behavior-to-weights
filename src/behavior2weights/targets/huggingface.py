from __future__ import annotations
from pathlib import Path
from typing import cast
import torch
from torch import nn
from behavior2weights.schemas import TargetRecord
from behavior2weights.targets.base import TargetAdapter,resolve_checkpoint
class HuggingFaceCausalLMAdapter(TargetAdapter):
    family="huggingface-causal-lm"
    def __init__(self,*,manifest_root:str|Path|None=None,local_files_only:bool=False)->None:
        self.manifest_root=manifest_root
        self.local_files_only=local_files_only
    def can_load(self,record:TargetRecord)->bool:
        return record.family_id==self.family
    def load(self,record:TargetRecord,*,device:str|torch.device="cpu")->nn.Module:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("Install behavior2weights[hf] to load Hugging Face models")from error
        path=resolve_checkpoint(record,manifest_root=self.manifest_root)
        source=str(path)if path.exists()else str(record.metadata.get("model_name",path))
        trust_remote_code=bool(record.metadata.get("trust_remote_code",False))
        revision=record.metadata.get("revision")
        dtype_name=str(record.metadata.get("torch_dtype","float32"))
        dtype=getattr(torch,dtype_name,torch.float32)
        model=AutoModelForCausalLM.from_pretrained(source,revision=revision,trust_remote_code=trust_remote_code,local_files_only=self.local_files_only,torch_dtype=dtype,)
        return cast(nn.Module,model.to(device))
