from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Any
from behavior2weights.schemas import Split,TargetRecord
from behavior2weights.utils import stablehash
from behavior2weights.zoo.manifest import savemanifest
@dataclasses.dataclass(frozen=True,slots=True)
class PublicModelSpec:
    model_name:str
    revisions:tuple[str,...]=("main",)
    external_family:str="unknown"
    architecture_id:str|None=None
    parameter_count:int|None=None
    tokenizer_name:str|None=None
    license:str|None=None
    trust_remote_code:bool=False
    torch_dtype:str="float32"
    split:Split=Split.OOD
    metadata:dict[str,Any]=dataclasses.field(default_factory=dict)
@dataclasses.dataclass(frozen=True,slots=True)
class PublicManifestConfig:
    models:tuple[PublicModelSpec,...]
    @classmethod
    def fromdict(cls,raw:dict[str,Any])->PublicManifestConfig:
        models:list[PublicModelSpec]=[]
        for item in raw["models"]:
            value=dict(item)
            value["revisions"]=tuple(value.get("revisions",["main"]))
            value["split"]=Split(value.get("split","ood"))
            models.append(PublicModelSpec(**value))
        return cls(models=tuple(models))
def buildpublicmanifest(config:PublicManifestConfig,output_path:str|Path)->list[TargetRecord]:
    records:list[TargetRecord]=[]
    for model in config.models:
        architecture_id=model.architecture_id or f"hf-{stablehash(model.model_name)}"
        lineage_id=(f"public-{stablehash({'model': model.model_name, 'family': model.external_family})}")
        for revision in model.revisions:
            target_id=f"public-{stablehash({'model': model.model_name, 'revision': revision})}"
            step=None
            if revision.startswith("step")and revision[4:].isdigit():
                step=int(revision[4:])
            records.append(TargetRecord(target_id=target_id,family_id="huggingface-causal-lm",lineage_id=lineage_id,architecture_id=architecture_id,task_id="public-pretraining",dataset_id=str(model.metadata.get("dataset_id","unknown")),seed=int(model.metadata.get("seed",0)),checkpoint_path=Path(model.model_name),training_step=step,split=model.split,factors={"external_family":model.external_family,"parameter_count":model.parameter_count,"revision":revision,"model_name":model.model_name,},metadata={**model.metadata,"model_name":model.model_name,"revision":revision,"tokenizer_name":model.tokenizer_name or model.model_name,"license":model.license,"trust_remote_code":model.trust_remote_code,"torch_dtype":model.torch_dtype,},))
    savemanifest(records,output_path)
    return records
