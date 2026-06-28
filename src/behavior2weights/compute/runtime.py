from __future__ import annotations
import dataclasses
import os
import platform
import sys
from typing import Any
import torch
@dataclasses.dataclass(frozen=True,slots=True)
class RuntimeInfo:
    python:str
    platform:str
    torch:str
    cuda_available:bool
    cuda_version:str|None
    gpu_count:int
    gpu_names:tuple[str,...]
    environment:dict[str,str]
def resolvedevice(requested:str="auto")->torch.device:
    if requested!="auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends,"mps")and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
def runtimeinfo()->RuntimeInfo:
    gpu_names=tuple(torch.cuda.get_device_name(index)for index in range(torch.cuda.device_count()))
    tracked_environment={key:os.environ[key]for key in("CUDA_VISIBLE_DEVICES","SLURM_JOB_ID","SLURM_ARRAY_TASK_ID","AWS_REGION","AIM_REPO",)if key in os.environ}
    return RuntimeInfo(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_version=torch.version.cuda,gpu_count=torch.cuda.device_count(),gpu_names=gpu_names,environment=tracked_environment,)
def runtimeinfodict()->dict[str,Any]:
    return dataclasses.asdict(runtimeinfo())
