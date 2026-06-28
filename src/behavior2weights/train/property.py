from __future__ import annotations
import dataclasses
import json
import math
from pathlib import Path
from typing import Any
import torch
from safetensors.torch import load_file,save_file
from torch import Tensor
from torch.nn import functional as F
from behavior2weights.models.inverse import CHANNEL_TO_ID
from behavior2weights.models.property import BehaviorPropertyClassifier,PropertyModelConfig
from behavior2weights.schemas import Split,TargetRecord
from behavior2weights.traces.store import TraceBundle
from behavior2weights.tracking.base import ExperimentTracker
from behavior2weights.utils import seedeverything
UNKNOWN="__UNKNOWN__"
def propertyvalue(record:TargetRecord,name:str)->str:
    if name=="architecture_id":
        return record.architecture_id
    if name=="family_id":
        return record.family_id
    if name=="task_id":
        return record.task_id
    if name=="dataset_id":
        return record.dataset_id
    if name=="intervention_kind":
        return record.interventions[0].kind if record.interventions else "none"
    if name=="has_intervention":
        return str(bool(record.interventions)).lower()
    if name=="training_step_bin":
        step=record.training_step
        if step is None:
            return "unknown"
        if step==0:
            return "step0"
        if step<=128:
            return "early"
        if step<=10_000:
            return "middle"
        return "late"
    if name.startswith("factor:"):
        return str(record.factors.get(name.split(":",1)[1],"missing"))
    if name.startswith("metadata:"):
        return str(record.metadata.get(name.split(":",1)[1],"missing"))
    raise KeyError(f"unknown property specification: {name}")
@dataclasses.dataclass(frozen=True,slots=True)
class PropertyVocabulary:
    values:dict[str,tuple[str,...]]
    @classmethod
    def fit(cls,records:list[TargetRecord],properties:tuple[str,...])->PropertyVocabulary:
        values:dict[str,tuple[str,...]]={}
        for name in properties:
            observed=sorted({propertyvalue(record,name)for record in records})
            values[name]=tuple([UNKNOWN,*observed])
        return cls(values)
    def encode(self,record:TargetRecord,name:str)->int:
        value=propertyvalue(record,name)
        values=self.values[name]
        return values.index(value)if value in values else 0
    def decode(self,name:str,index:int)->str:
        return self.values[name][index]
    @property
    def dimensions(self)->dict[str,int]:
        return{name:len(values)for name,values in self.values.items()}
    def save(self,path:str|Path)->None:
        Path(path).write_text(json.dumps(self.values,indent=2,sort_keys=True)+"\n")
    @classmethod
    def load(cls,path:str|Path)->PropertyVocabulary:
        raw=json.loads(Path(path).read_text())
        return cls({name:tuple(values)for name,values in raw.items()})
class TracePropertyCorpus:
    def __init__(self,records:list[TargetRecord],traces:TraceBundle,*,properties:tuple[str,...],vocabulary:PropertyVocabulary|None=None,)->None:
        trace_index={target_id:index for index,target_id in enumerate(traces.target_ids)}
        self.records=[record for record in records if record.target_id in trace_index]
        if not self.records:
            raise ValueError("no manifest records match trace targets")
        self.traces=traces
        self.trace_indices=torch.tensor([trace_index[record.target_id]for record in self.records])
        self.properties=properties
        training_records=[record for record in self.records if record.split==Split.TRAIN]
        if not training_records:
            raise ValueError("property corpus contains no training records")
        self.vocabulary=vocabulary or PropertyVocabulary.fit(training_records,properties)
    def indicesforsplit(self,split:Split)->list[int]:
        return[index for index,record in enumerate(self.records)if record.split==split]
    def batch(self,target_indices:list[int],*,query_indices:Tensor,)->dict[str,Any]:
        if not target_indices:
            raise ValueError("target_indices cannot be empty")
        query_indices=query_indices.long()
        rows=self.trace_indices[torch.tensor(target_indices)]
        input_ids=self.traces.input_ids[query_indices].unsqueeze(0).expand(len(rows),-1,-1)
        observations=self.traces.observations[rows][:,query_indices]
        channel_ids=torch.full((len(rows),len(query_indices)),CHANNEL_TO_ID[self.traces.channel],dtype=torch.long,)
        labels={name:torch.tensor([self.vocabulary.encode(self.records[index],name)for index in target_indices],dtype=torch.long,)for name in self.properties}
        return{"input_ids":input_ids,"observations":observations,"channel_ids":channel_ids,"query_mask":torch.ones(len(rows),len(query_indices),dtype=torch.bool),"labels":labels,}
@dataclasses.dataclass(frozen=True,slots=True)
class PropertyTrainingConfig:
    properties:tuple[str,...]=("architecture_id","task_id","intervention_kind","training_step_bin",)
    steps:int=2_000
    batch_size:int=32
    query_budgets:tuple[int,...]=(16,32,64)
    learning_rate:float=3e-4
    weight_decay:float=1e-4
    warmup_steps:int=100
    validation_every:int=100
    validation_batches:int=8
    early_stopping_patience:int=10
    seed:int=20260621
    device:str="cpu"
    @classmethod
    def fromdict(cls,raw:dict[str,Any])->PropertyTrainingConfig:
        known={field.name for field in dataclasses.fields(cls)}
        unknown=set(raw)-known
        if unknown:
            raise ValueError(f"Unknown PropertyTrainingConfig fields: {sorted(unknown)}")
        data=dict(raw)
        if "properties" in data:
            data["properties"]=tuple(str(value)for value in data["properties"])
        if "query_budgets" in data:
            data["query_budgets"]=tuple(int(value)for value in data["query_budgets"])
        return cls(**data)
    def __post_init__(self)->None:
        if not self.properties:
            raise ValueError("properties cannot be empty")
        if self.steps<=0 or self.batch_size<=0:
            raise ValueError("steps and batch_size must be positive")
        if not self.query_budgets or min(self.query_budgets)<=0:
            raise ValueError("query_budgets must contain positive values")
        if self.validation_every<=0 or self.validation_batches<=0:
            raise ValueError("validation settings must be positive")
        if self.early_stopping_patience<=0:
            raise ValueError("early_stopping_patience must be positive")
@dataclasses.dataclass(frozen=True,slots=True)
class PropertyTrainResult:
    best_checkpoint:Path
    best_validation_loss:float
    steps_completed:int
def _move(batch:dict[str,Any],device:torch.device)->dict[str,Any]:
    return{key:({name:value.to(device)for name,value in item.items()}if isinstance(item,dict)else item.to(device))for key,item in batch.items()}
def _loss(logits:dict[str,Tensor],labels:dict[str,Tensor])->Tensor:
    return torch.stack([F.cross_entropy(logits[name],labels[name])for name in logits]).mean()
def lrscale(step:int,config:PropertyTrainingConfig)->float:
    if step<config.warmup_steps:
        return(step+1)/max(config.warmup_steps,1)
    progress=(step-config.warmup_steps)/max(config.steps-config.warmup_steps,1)
    return 0.5*(1+math.cos(math.pi*min(max(progress,0),1)))
def savepropertycheckpoint(model:BehaviorPropertyClassifier,vocabulary:PropertyVocabulary,directory:str|Path,metadata:dict[str,Any],)->Path:
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/"property_model.safetensors"
    save_file({name:value.detach().cpu()for name,value in model.state_dict().items()},str(path))
    (directory/"model_config.json").write_text(json.dumps(model.config.todict(),indent=2,sort_keys=True)+"\n")
    vocabulary.save(directory/"vocabulary.json")
    (directory/"metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    return path
def loadpropertycheckpoint(directory:str|Path,*,device:str="cpu")->tuple[BehaviorPropertyClassifier,PropertyVocabulary]:
    directory=Path(directory)
    config=PropertyModelConfig.fromdict(json.loads((directory/"model_config.json").read_text()))
    model=BehaviorPropertyClassifier(config)
    model.load_state_dict(load_file(str(directory/"property_model.safetensors"),device=device))
    return model.to(device),PropertyVocabulary.load(directory/"vocabulary.json")
def trainpropertyclassifier(model_config:PropertyModelConfig,corpus:TracePropertyCorpus,config:PropertyTrainingConfig,output_directory:str|Path,*,tracker:ExperimentTracker|None=None,)->PropertyTrainResult:
    seedeverything(config.seed)
    device=torch.device(config.device)
    if model_config.property_dims!=corpus.vocabulary.dimensions:
        raise ValueError("model property dimensions do not match fitted vocabulary")
    if max(config.query_budgets)>len(corpus.traces.query_ids):
        raise ValueError("a property-training query budget exceeds the trace bank")
    output_directory=Path(output_directory)
    output_directory.mkdir(parents=True,exist_ok=True)
    model=BehaviorPropertyClassifier(model_config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    train_indices=corpus.indicesforsplit(Split.TRAIN)
    validation_indices=corpus.indicesforsplit(Split.VALIDATION)or corpus.indicesforsplit(Split.TEST)
    if not validation_indices:
        validation_indices=train_indices[:min(8,len(train_indices))]
    generator=torch.Generator().manual_seed(config.seed+1)
    best_loss=float("inf")
    stale=0
    best_dir=output_directory/"best"
    history:list[dict[str,float]]=[]
    if tracker:
        tracker.setparams({"training":dataclasses.asdict(config),"model":model_config.todict(),"property_vocabulary":corpus.vocabulary.values,"train_targets":len(train_indices),"validation_targets":len(validation_indices),"channel":corpus.traces.channel.value,})
    try:
        for step in range(config.steps):
            model.train()
            budget=config.query_budgets[step%len(config.query_budgets)]
            selected=[train_indices[int(value)]for value in torch.randint(len(train_indices),(config.batch_size,),generator=generator)]
            queries=torch.randperm(len(corpus.traces.query_ids),generator=generator)[:budget]
            batch=_move(corpus.batch(selected,query_indices=queries),device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch["input_ids"],batch["observations"],batch["channel_ids"],query_mask=batch["query_mask"],)
            loss=_loss(logits,batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scale=lrscale(step,config)
            for group in optimizer.param_groups:
                group["lr"]=config.learning_rate*scale
            optimizer.step()
            row={"step":float(step),"loss":float(loss.item()),"query_budget":float(budget),"learning_rate":float(config.learning_rate*scale),}
            history.append(row)
            if tracker:
                tracker.track(row["loss"],name="property_loss",step=step,context={"subset":"train"})
                tracker.track(row["learning_rate"],name="learning_rate",step=step,context={"subset":"train"},)
            should_validate=(step+1)%config.validation_every==0 or step+1==config.steps
            if not should_validate:
                continue
            model.eval()
            losses:list[float]=[]
            with torch.no_grad():
                for repeat in range(config.validation_batches):
                    qgen=torch.Generator().manual_seed(config.seed+100_000+step*31+repeat)
                    chosen=[validation_indices[int(value)]for value in torch.randint(len(validation_indices),(min(config.batch_size,len(validation_indices)),),generator=qgen,)]
                    q=torch.randperm(len(corpus.traces.query_ids),generator=qgen)[:max(config.query_budgets)]
                    validation_batch=_move(corpus.batch(chosen,query_indices=q),device)
                    validation_logits=model(validation_batch["input_ids"],validation_batch["observations"],validation_batch["channel_ids"],query_mask=validation_batch["query_mask"],)
                    losses.append(float(_loss(validation_logits,validation_batch["labels"]).item()))
            validation_loss=sum(losses)/len(losses)
            history[-1]["validation_loss"]=validation_loss
            if tracker:
                tracker.track(validation_loss,name="property_loss",step=step,context={"subset":"validation"},)
            if validation_loss<best_loss:
                best_loss=validation_loss
                stale=0
                savepropertycheckpoint(model,corpus.vocabulary,best_dir,{"validation_loss":best_loss,"step":step,"history_tail":history[-20:],},)
            else:
                stale+=1
                if stale>=config.early_stopping_patience:
                    break
        if not best_dir.exists():
            savepropertycheckpoint(model,corpus.vocabulary,best_dir,{"validation_loss":best_loss,"step":len(history)-1},)
        (output_directory/"training_history.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in history))
        if tracker:
            tracker.close("completed")
    except Exception:
        if tracker:
            tracker.close("failed")
        raise
    return PropertyTrainResult(best_dir,best_loss,len(history))
