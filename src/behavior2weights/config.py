from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel,ConfigDict,Field,model_validator
from behavior2weights.utils import deepmerge,loadyaml
class StrictModel(BaseModel):
    model_config=ConfigDict(extra="forbid",validate_assignment=True)
class PathsConfig(StrictModel):
    data_root:Path=Path("data")
    artifact_root:Path=Path("artifacts")
    cache_root:Path=Path("data/cache")
class TrackingConfig(StrictModel):
    backend:str="jsonl"
    repo:str="./aim-data"
    experiment:str="behavior-to-weights"
    enabled:bool=True
    tags:list[str]=Field(default_factory=list)
    system_tracking_interval:float|None=10.0
class ReproducibilityConfig(StrictModel):
    seed:int=20260621
    deterministic:bool=True
    save_environment:bool=True
class RuntimeConfig(StrictModel):
    device:str="auto"
    dtype:str="float32"
    num_workers:int=0
    compile:bool=False
class RootConfig(StrictModel):
    name:str="experiment"
    paths:PathsConfig=Field(default_factory=PathsConfig)
    tracking:TrackingConfig=Field(default_factory=TrackingConfig)
    reproducibility:ReproducibilityConfig=Field(default_factory=ReproducibilityConfig)
    runtime:RuntimeConfig=Field(default_factory=RuntimeConfig)
    experiment:dict[str,Any]=Field(default_factory=dict)
    @model_validator(mode="after")
    def ensurepaths(self)->RootConfig:
        self.paths.data_root=Path(self.paths.data_root)
        self.paths.artifact_root=Path(self.paths.artifact_root)
        self.paths.cache_root=Path(self.paths.cache_root)
        return self
def loadrootconfig(path:str|Path,overrides:dict[str,Any]|None=None)->RootConfig:
    raw=loadyaml(path)
    if overrides:
        raw=deepmerge(raw,overrides)
    return RootConfig.model_validate(raw)
