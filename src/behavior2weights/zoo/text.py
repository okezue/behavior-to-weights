from __future__ import annotations
import dataclasses
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
from torch.nn import functional as F
from behavior2weights.models.microtransformer import MicroTransformer,MicroTransformerConfig
from behavior2weights.schemas import TargetRecord
from behavior2weights.utils import seedeverything,stablehash
from behavior2weights.zoo.manifest import SplitPolicy,assignlineagesplits,savemanifest
from behavior2weights.zoo.micro import(InterventionSpec,OptimizerSpec,ZooBuildResult,interventiontargets,makeoptimizer,savetarget,)
@dataclasses.dataclass(frozen=True,slots=True)
class TextZooConfig:
    architectures:tuple[MicroTransformerConfig,...]
    model_seeds:tuple[int,...]=tuple(range(8))
    data_order_seeds:tuple[int,...]=(0,)
    optimizers:tuple[OptimizerSpec,...]=(OptimizerSpec(),)
    train_steps:int=20_000
    checkpoint_steps:tuple[int,...]=(0,1_000,5_000,20_000)
    batch_size:int=64
    validation_batches:int=32
    gradient_clip:float=1.0
    interventions:tuple[InterventionSpec,...]=()
    deterministic:bool=True
    @classmethod
    def fromdict(cls,raw:Mapping[str,Any])->TextZooConfig:
        known={field.name for field in dataclasses.fields(cls)}
        unknown=set(raw)-known
        if unknown:
            raise ValueError(f"Unknown TextZooConfig fields: {sorted(unknown)}")
        data=dict(raw)
        data["architectures"]=tuple(MicroTransformerConfig.fromdict(value)for value in data["architectures"])
        data["model_seeds"]=tuple(int(value)for value in _expand(data.get("model_seeds",range(8))))
        data["data_order_seeds"]=tuple(int(value)for value in _expand(data.get("data_order_seeds",[0])))
        data["checkpoint_steps"]=tuple(int(value)for value in data.get("checkpoint_steps",[0,data.get("train_steps",20_000)]))
        data["optimizers"]=tuple(OptimizerSpec(**value)for value in data.get("optimizers",[dataclasses.asdict(OptimizerSpec())]))
        data["interventions"]=tuple(InterventionSpec(**value)for value in data.get("interventions",[]))
        return cls(**data)
    def __post_init__(self)->None:
        if not self.architectures:
            raise ValueError("at least one architecture is required")
        if not self.model_seeds or not self.data_order_seeds:
            raise ValueError("model_seeds and data_order_seeds cannot be empty")
        if self.train_steps<0:
            raise ValueError("train_steps cannot be negative")
        if self.train_steps not in self.checkpoint_steps:
            raise ValueError("checkpoint_steps must include train_steps")
        if any(step<0 or step>self.train_steps for step in self.checkpoint_steps):
            raise ValueError("checkpoint steps must lie in [0, train_steps]")
        if len(set(self.checkpoint_steps))!=len(self.checkpoint_steps):
            raise ValueError("checkpoint_steps must be unique")
        if self.batch_size<=0 or self.validation_batches<=0:
            raise ValueError("batch_size and validation_batches must be positive")
        if self.gradient_clip<=0:
            raise ValueError("gradient_clip must be positive")
def _expand(value:Any)->tuple[int,...]:
    if isinstance(value,Mapping):
        return tuple(range(int(value.get("start",0)),int(value["stop"]),int(value.get("step",1))))
    return tuple(int(item)for item in value)
class ArrowBatchStream:
    def __init__(self,dataset:Any,batch_size:int,seed:int)->None:
        if len(dataset)==0:
            raise ValueError("training dataset is empty")
        self.dataset=dataset.with_format("torch",columns=["input_ids"])
        self.batch_size=batch_size
        self.generator=torch.Generator().manual_seed(seed)
        self.order=torch.randperm(len(dataset),generator=self.generator)
        self.position=0
    def next(self)->torch.Tensor:
        if self.position+self.batch_size>len(self.order):
            self.order=torch.randperm(len(self.dataset),generator=self.generator)
            self.position=0
        indices=self.order[self.position:self.position+self.batch_size].tolist()
        self.position+=self.batch_size
        batch=self.dataset[indices]["input_ids"]
        return batch if isinstance(batch,torch.Tensor)else torch.tensor(batch,dtype=torch.long)
@torch.no_grad()
def validationloss(model:MicroTransformer,dataset:Any,batches:int,batch_size:int,device:str,)->float:
    model.eval()
    formatted=dataset.with_format("torch",columns=["input_ids"])
    losses:list[float]=[]
    for start in range(0,min(len(formatted),batches*batch_size),batch_size):
        sequence=formatted[list(range(start,min(start+batch_size,len(formatted))))]["input_ids"]
        if not isinstance(sequence,torch.Tensor):
            sequence=torch.tensor(sequence,dtype=torch.long)
        sequence=sequence.to(device)
        logits=model(sequence[:,:-1])
        losses.append(float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),sequence[:,1:].reshape(-1)).item()))
    if not losses:
        raise ValueError("validation dataset contains no usable examples")
    return float(sum(losses)/len(losses))
def buildtextzoo(config:TextZooConfig,dataset_directory:str|Path,output_root:str|Path,*,device:str="cpu",split_policy:SplitPolicy|None=None,)->ZooBuildResult:
    split_policy=split_policy or SplitPolicy()
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError("Install behavior2weights[hf] to build a text model zoo")from error
    dataset_directory=Path(dataset_directory)
    manifest=json.loads((dataset_directory/"dataset_manifest.json").read_text())
    datasets=load_from_disk(str(dataset_directory/"arrow"))
    if "train" not in datasets:
        raise ValueError("prepared dataset has no train split")
    validation_name=("validation" if "validation" in datasets else("test" if "test" in datasets else None))
    if validation_name is None:
        raise ValueError("prepared dataset needs validation or test split")
    tokenizer=manifest["tokenizer"]
    expected_vocab=int(tokenizer["vocab_size"])
    expected_context=int(manifest["config"]["sequence_length"])
    dataset_id=f"hf-{stablehash({'fingerprints': manifest['dataset_fingerprint'], 'tokenizer': tokenizer}, length=24)}"
    root=Path(output_root)
    root.mkdir(parents=True,exist_ok=True)
    records:list[TargetRecord]=[]
    metrics:list[dict[str,Any]]=[]
    combinations=itertools.product(enumerate(config.architectures),config.model_seeds,config.data_order_seeds,config.optimizers,)
    checkpoint_steps=set(config.checkpoint_steps)
    for(architecture_index,architecture),model_seed,order_seed,optimizer_spec in combinations:
        if architecture.vocab_size!=expected_vocab:
            raise ValueError(f"architecture vocab_size={architecture.vocab_size} but tokenizer has {expected_vocab} tokens")
        if architecture.max_seq_len!=expected_context:
            raise ValueError(f"architecture max_seq_len={architecture.max_seq_len} but dataset context is {expected_context}")
        architecture_dict=dataclasses.asdict(architecture)
        architecture_id=f"micro-text-{stablehash(architecture_dict)}"
        lineage_payload={"architecture":architecture_dict,"dataset_id":dataset_id,"model_seed":model_seed,"data_order_seed":order_seed,"optimizer":dataclasses.asdict(optimizer_spec),}
        lineage_id=f"text-lin-{stablehash(lineage_payload)}"
        seedeverything(model_seed,deterministic=config.deterministic)
        model=MicroTransformer(architecture).to(device)
        optimizer=makeoptimizer(model,optimizer_spec)
        stream=ArrowBatchStream(datasets["train"],config.batch_size,order_seed)
        final_record:TargetRecord|None=None
        for step in range(config.train_steps+1):
            if step in checkpoint_steps:
                target_id=f"target-{stablehash({'lineage': lineage_id, 'step': step})}"
                relative=Path("checkpoints")/lineage_id/f"step-{step:08d}.safetensors"
                path,checksum=savetarget(model,root=root,relative_path=relative,metadata={"target_id":target_id,"lineage_id":lineage_id,"training_step":str(step),},)
                validation_loss=validationloss(model,datasets[validation_name],config.validation_batches,config.batch_size,device,)
                record=TargetRecord(target_id=target_id,family_id="micro-transformer",lineage_id=lineage_id,architecture_id=architecture_id,task_id="causal-language-modeling",dataset_id=dataset_id,seed=model_seed,checkpoint_path=path,checkpoint_sha256=checksum,training_step=step,factors={"architecture_index":architecture_index,"d_model":architecture.d_model,"n_layers":architecture.n_layers,"n_heads":architecture.n_heads,"d_ff":architecture.d_ff,"model_seed":model_seed,"data_order_seed":order_seed,"optimizer":optimizer_spec.name,"learning_rate":optimizer_spec.learning_rate,"training_step":step,"intervention_kind":"none",},metadata={"model_config":architecture_dict,"optimizer":dataclasses.asdict(optimizer_spec),"dataset_manifest":manifest,"validation_loss":validation_loss,"num_parameters":model.numparameters(),},)
                records.append(record)
                metrics.append({"target_id":target_id,"lineage_id":lineage_id,"step":step,"validation_loss":validation_loss,})
                if step==config.train_steps:
                    final_record=record
            if step==config.train_steps:
                break
            sequence=stream.next().to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits=model(sequence[:,:-1])
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),sequence[:,1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip)
            optimizer.step()
        if final_record and config.interventions:
            records.extend(interventiontargets(model,final_record,config.interventions,root=root,lineage_seed=model_seed*1_000_003+order_seed,))
    records=assignlineagesplits(records,split_policy)
    savemanifest(records,root/"targets.jsonl")
    (root/"training_metrics.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in metrics))
    (root/"source_dataset.json").write_text(json.dumps({"dataset_directory":str(dataset_directory),"dataset_id":dataset_id,"manifest":manifest,},indent=2,sort_keys=True,)+"\n")
    return ZooBuildResult(root/"targets.jsonl",tuple(records),tuple(metrics))
