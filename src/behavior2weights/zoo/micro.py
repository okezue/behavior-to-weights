from __future__ import annotations
import dataclasses
import itertools
from collections.abc import Mapping,Sequence
from pathlib import Path
from typing import Any
import torch
from torch.nn import functional as F
from behavior2weights.data.synthetic import SequenceBatcher,SyntheticDatasetConfig,build_splits
from behavior2weights.models.micro_transformer import MicroTransformer,MicroTransformerConfig
from behavior2weights.schemas import TargetRecord
from behavior2weights.utils import file_sha256,seed_everything,stable_hash
from behavior2weights.zoo.interventions import(ablate_attention_head,ablate_mlp_neuron,apply_lora_edit,apply_sparse_weight_edit,)
from behavior2weights.zoo.manifest import SplitPolicy,assign_lineage_splits,save_manifest
def _expand_int_values(value:Any)->tuple[int,...]:
    if isinstance(value,Mapping):
        start=int(value.get("start",0))
        stop=int(value["stop"])
        step=int(value.get("step",1))
        return tuple(range(start,stop,step))
    return tuple(int(item)for item in value)
@dataclasses.dataclass(frozen=True,slots=True)
class OptimizerSpec:
    name:str="adamw"
    learning_rate:float=3e-3
    weight_decay:float=0.0
    beta1:float=0.9
    beta2:float=0.95
@dataclasses.dataclass(frozen=True,slots=True)
class InterventionSpec:
    kind:str
    count:int=1
    scale:float=0.1
    rank:int=2
    tensor_name:str|None=None
@dataclasses.dataclass(frozen=True,slots=True)
class MicroZooConfig:
    architectures:tuple[MicroTransformerConfig,...]
    tasks:tuple[str,...]=("mixture",)
    model_seeds:tuple[int,...]=tuple(range(8))
    dataset_seeds:tuple[int,...]=(0,)
    optimizers:tuple[OptimizerSpec,...]=(OptimizerSpec(),)
    train_steps:int=200
    checkpoint_steps:tuple[int,...]=(0,50,200)
    batch_size:int=64
    train_examples:int=2_048
    validation_examples:int=256
    test_examples:int=256
    gradient_clip:float=1.0
    interventions:tuple[InterventionSpec,...]=()
    deterministic:bool=True
    def __post_init__(self)->None:
        if not self.architectures:
            raise ValueError("at least one architecture is required")
        if self.train_steps<0:
            raise ValueError("train_steps cannot be negative")
        if any(step<0 or step>self.train_steps for step in self.checkpoint_steps):
            raise ValueError("checkpoint_steps must be within [0, train_steps]")
        if self.train_steps not in self.checkpoint_steps:
            raise ValueError("checkpoint_steps must include train_steps")
    @classmethod
    def from_dict(cls,raw:Mapping[str,Any])->MicroZooConfig:
        known={field.name for field in dataclasses.fields(cls)}
        unknown=set(raw)-known
        if unknown:
            raise ValueError(f"Unknown MicroZooConfig fields: {sorted(unknown)}")
        data=dict(raw)
        data["architectures"]=tuple(MicroTransformerConfig.from_dict(item)for item in data["architectures"])
        data["optimizers"]=tuple(OptimizerSpec(**item)for item in data.get("optimizers",[dataclasses.asdict(OptimizerSpec())]))
        data["interventions"]=tuple(InterventionSpec(**item)for item in data.get("interventions",[]))
        if "tasks" in data:
            data["tasks"]=tuple(data["tasks"])
        for key in("model_seeds","dataset_seeds","checkpoint_steps"):
            if key in data:
                data[key]=_expand_int_values(data[key])
        return cls(**data)
@dataclasses.dataclass(frozen=True,slots=True)
class ZooBuildResult:
    manifest_path:Path
    records:tuple[TargetRecord,...]
    training_metrics:tuple[dict[str,Any],...]
def _make_optimizer(model:MicroTransformer,spec:OptimizerSpec)->torch.optim.Optimizer:
    if spec.name.lower()=="adamw":
        return torch.optim.AdamW(model.parameters(),lr=spec.learning_rate,weight_decay=spec.weight_decay,betas=(spec.beta1,spec.beta2),)
    if spec.name.lower()=="sgd":
        return torch.optim.SGD(model.parameters(),lr=spec.learning_rate,weight_decay=spec.weight_decay,momentum=spec.beta1,)
    raise ValueError(f"Unsupported optimizer: {spec.name}")
@torch.no_grad()
def evaluate_loss(model:MicroTransformer,sequences:torch.Tensor,batch_size:int=128)->float:
    model.eval()
    losses:list[float]=[]
    weights:list[int]=[]
    for start in range(0,len(sequences),batch_size):
        batch=sequences[start:start+batch_size]
        logits=model(batch[:,:-1])
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch[:,1:].reshape(-1))
        losses.append(float(loss.item()))
        weights.append(len(batch))
    return float(sum(loss*weight for loss,weight in zip(losses,weights,strict=True))/sum(weights))
def _save_target(model:MicroTransformer,*,root:Path,relative_path:Path,metadata:dict[str,str],)->tuple[Path,str]:
    absolute=root/relative_path
    model.save(absolute,metadata=metadata)
    return relative_path,file_sha256(absolute)
def _intervention_targets(base_model:MicroTransformer,base_record:TargetRecord,specs:Sequence[InterventionSpec],*,root:Path,lineage_seed:int,)->list[TargetRecord]:
    records:list[TargetRecord]=[]
    config=base_model.config
    base_state=base_model.state_dict()
    counter=0
    for spec in specs:
        for local_index in range(spec.count):
            intervention_seed=lineage_seed+counter*104_729+local_index
            if spec.kind=="attention_head_ablation":
                layer=intervention_seed%config.n_layers
                head=(intervention_seed//max(config.n_layers,1))%config.n_heads
                result=ablate_attention_head(base_state,config,layer=layer,head=head)
            elif spec.kind=="mlp_neuron_ablation":
                layer=intervention_seed%config.n_layers
                neuron=(intervention_seed//max(config.n_layers,1))%config.d_ff
                result=ablate_mlp_neuron(base_state,config,layer=layer,neuron=neuron)
            elif spec.kind=="lora_edit":
                tensor_name=spec.tensor_name or "blocks.0.attn.o_proj.weight"
                result=apply_lora_edit(base_state,tensor_name=tensor_name,rank=min(spec.rank,min(base_state[tensor_name].shape)),scale=spec.scale,seed=intervention_seed,)
            elif spec.kind=="sparse_weight_edit":
                tensor_name=spec.tensor_name or "blocks.0.mlp.fc2.weight"
                count=min(max(1,spec.rank),base_state[tensor_name].numel())
                result=apply_sparse_weight_edit(base_state,tensor_name=tensor_name,count=count,scale=spec.scale,seed=intervention_seed,)
            else:
                raise ValueError(f"Unknown intervention kind: {spec.kind}")
            edited=MicroTransformer(config)
            edited.load_state_dict(result.state_dict,strict=True)
            edit_id=stable_hash({"parent":base_record.target_id,"intervention":result.record.model_dump(mode="json"),})
            relative_path=(Path("checkpoints")/base_record.lineage_id/f"edit-{edit_id}.safetensors")
            path,checksum=_save_target(edited,root=root,relative_path=relative_path,metadata={"target_id":edit_id,"parent_target_id":base_record.target_id},)
            records.append(base_record.model_copy(update={"target_id":edit_id,"checkpoint_path":path,"checkpoint_sha256":checksum,"parent_target_id":base_record.target_id,"interventions":[result.record],"factors":{**base_record.factors,"intervention_kind":result.record.kind,"intervention_label":result.record.label,},"metadata":{**base_record.metadata,"changed_tensors":list(result.changed_tensors),"changed_entries":result.changed_entries,},}))
            counter+=1
    return records
def build_micro_zoo(config:MicroZooConfig,output_root:str|Path,*,device:str|torch.device="cpu",split_policy:SplitPolicy|None=None,resume:bool=True,)->ZooBuildResult:
    split_policy=split_policy or SplitPolicy()
    root=Path(output_root)
    root.mkdir(parents=True,exist_ok=True)
    records:list[TargetRecord]=[]
    metrics:list[dict[str,Any]]=[]
    combinations=itertools.product(enumerate(config.architectures),config.tasks,config.model_seeds,config.dataset_seeds,config.optimizers,)
    for(architecture_index,architecture,),task,model_seed,dataset_seed,optimizer_spec in combinations:
        architecture_dict=dataclasses.asdict(architecture)
        architecture_id=f"micro-{stable_hash(architecture_dict)}"
        lineage_payload={"architecture":architecture_dict,"task":task,"model_seed":model_seed,"dataset_seed":dataset_seed,"optimizer":dataclasses.asdict(optimizer_spec),}
        lineage_id=f"lin-{stable_hash(lineage_payload)}"
        seed_everything(model_seed,deterministic=config.deterministic)
        model=MicroTransformer(architecture).to(device)
        optimizer=_make_optimizer(model,optimizer_spec)
        dataset_config=SyntheticDatasetConfig(task=task,vocab_size=architecture.vocab_size,seq_len=architecture.max_seq_len,train_examples=config.train_examples,validation_examples=config.validation_examples,test_examples=config.test_examples,seed=dataset_seed,)
        splits=build_splits(dataset_config)
        batcher=SequenceBatcher(splits["train"],config.batch_size,shuffle=True,seed=model_seed+dataset_seed)
        checkpoint_steps=set(config.checkpoint_steps)
        epoch=0
        step=0
        base_final_record:TargetRecord|None=None
        while step<=config.train_steps:
            if step in checkpoint_steps:
                target_id=f"target-{stable_hash({'lineage': lineage_id, 'step': step})}"
                relative_path=Path("checkpoints")/lineage_id/f"step-{step:08d}.safetensors"
                absolute_path=root/relative_path
                if not(resume and absolute_path.exists()):
                    path,checksum=_save_target(model,root=root,relative_path=relative_path,metadata={"target_id":target_id,"lineage_id":lineage_id,"training_step":str(step),},)
                else:
                    path,checksum=relative_path,file_sha256(absolute_path)
                validation_loss=evaluate_loss(model,splits["validation"])
                record=TargetRecord(target_id=target_id,family_id="micro-transformer",lineage_id=lineage_id,architecture_id=architecture_id,task_id=task,dataset_id=f"synthetic-{task}-seed{dataset_seed}",seed=model_seed,checkpoint_path=path,checkpoint_sha256=checksum,training_step=step,factors={"architecture_index":architecture_index,"d_model":architecture.d_model,"n_layers":architecture.n_layers,"n_heads":architecture.n_heads,"d_ff":architecture.d_ff,"task":task,"model_seed":model_seed,"dataset_seed":dataset_seed,"optimizer":optimizer_spec.name,"learning_rate":optimizer_spec.learning_rate,"training_step":step,"intervention_kind":"none",},metadata={"model_config":architecture_dict,"optimizer":dataclasses.asdict(optimizer_spec),"dataset_config":dataclasses.asdict(dataset_config),"validation_loss":validation_loss,"num_parameters":model.num_parameters(),},)
                records.append(record)
                metrics.append({"target_id":target_id,"lineage_id":lineage_id,"step":step,"validation_loss":validation_loss,})
                if step==config.train_steps:
                    base_final_record=record
            if step==config.train_steps:
                break
            for input_ids,labels in batcher.batches(epoch):
                model.train()
                input_ids=input_ids.to(device)
                labels=labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(input_ids)
                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip)
                optimizer.step()
                step+=1
                if step>=config.train_steps or step in checkpoint_steps:
                    break
            epoch+=1
        if base_final_record is None:
            raise RuntimeError("final checkpoint record was not generated")
        records.extend(_intervention_targets(model,base_final_record,config.interventions,root=root,lineage_seed=int(stable_hash(lineage_payload,length=8),16),))
    assigned=assign_lineage_splits(records,split_policy)
    manifest_path=root/"targets.jsonl"
    save_manifest(assigned,manifest_path)
    return ZooBuildResult(manifest_path=manifest_path,records=tuple(assigned),training_metrics=tuple(metrics),)
