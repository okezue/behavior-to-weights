from __future__ import annotations
import dataclasses
from collections import OrderedDict
from collections.abc import Mapping
from typing import Literal
import torch
from torch import Tensor
from behavior2weights.models.micro_transformer import MicroTransformerConfig
from behavior2weights.schemas import InterventionRecord
InterventionKind=Literal["attention_head_ablation","mlp_neuron_ablation","lora_edit","sparse_weight_edit",]
@dataclasses.dataclass(frozen=True,slots=True)
class InterventionResult:
    state_dict:OrderedDict[str,Tensor]
    record:InterventionRecord
    changed_tensors:tuple[str,...]
    changed_entries:int
def _clone_state(state_dict:Mapping[str,Tensor])->OrderedDict[str,Tensor]:
    return OrderedDict((name,value.detach().clone())for name,value in state_dict.items())
def ablate_attention_head(state_dict:Mapping[str,Tensor],config:MicroTransformerConfig,*,layer:int,head:int,)->InterventionResult:
    if not 0<=layer<config.n_layers:
        raise ValueError(f"layer must be in [0, {config.n_layers})")
    if not 0<=head<config.n_heads:
        raise ValueError(f"head must be in [0, {config.n_heads})")
    result=_clone_state(state_dict)
    start=head*config.head_dim
    end=(head+1)*config.head_dim
    prefix=f"blocks.{layer}.attn"
    changed:list[str]=[]
    entries=0
    for projection in("q_proj","k_proj","v_proj"):
        weight_name=f"{prefix}.{projection}.weight"
        result[weight_name][start:end,:]=0
        changed.append(weight_name)
        entries+=result[weight_name][start:end,:].numel()
        bias_name=f"{prefix}.{projection}.bias"
        if bias_name in result:
            result[bias_name][start:end]=0
            changed.append(bias_name)
            entries+=end-start
    output_name=f"{prefix}.o_proj.weight"
    result[output_name][:,start:end]=0
    changed.append(output_name)
    entries+=result[output_name][:,start:end].numel()
    return InterventionResult(state_dict=result,record=InterventionRecord(kind="attention_head_ablation",label=f"L{layer}.H{head}",layer=layer,head=head,metadata={"head_dim":config.head_dim},),changed_tensors=tuple(changed),changed_entries=entries,)
def ablate_mlp_neuron(state_dict:Mapping[str,Tensor],config:MicroTransformerConfig,*,layer:int,neuron:int,)->InterventionResult:
    if not 0<=layer<config.n_layers:
        raise ValueError(f"layer must be in [0, {config.n_layers})")
    if not 0<=neuron<config.d_ff:
        raise ValueError(f"neuron must be in [0, {config.d_ff})")
    result=_clone_state(state_dict)
    prefix=f"blocks.{layer}.mlp"
    fc1_weight=f"{prefix}.fc1.weight"
    fc1_bias=f"{prefix}.fc1.bias"
    fc2_weight=f"{prefix}.fc2.weight"
    result[fc1_weight][neuron,:]=0
    result[fc2_weight][:,neuron]=0
    changed=[fc1_weight,fc2_weight]
    entries=result[fc1_weight][neuron,:].numel()+result[fc2_weight][:,neuron].numel()
    if fc1_bias in result:
        result[fc1_bias][neuron]=0
        changed.append(fc1_bias)
        entries+=1
    return InterventionResult(state_dict=result,record=InterventionRecord(kind="mlp_neuron_ablation",label=f"L{layer}.N{neuron}",layer=layer,neuron=neuron,),changed_tensors=tuple(changed),changed_entries=entries,)
def apply_lora_edit(state_dict:Mapping[str,Tensor],*,tensor_name:str,rank:int,scale:float,seed:int,)->InterventionResult:
    result=_clone_state(state_dict)
    if tensor_name not in result:
        raise KeyError(tensor_name)
    weight=result[tensor_name]
    if weight.ndim!=2:
        raise ValueError("LoRA edits require a matrix-valued target tensor")
    if rank<=0 or rank>min(weight.shape):
        raise ValueError("rank must be positive and no larger than the smallest matrix dimension")
    generator=torch.Generator(device=weight.device).manual_seed(seed)
    left=torch.randn(weight.shape[0],rank,generator=generator,device=weight.device)
    right=torch.randn(rank,weight.shape[1],generator=generator,device=weight.device)
    left=left/left.norm(dim=0,keepdim=True).clamp_min(1e-8)
    right=right/right.norm(dim=1,keepdim=True).clamp_min(1e-8)
    update=scale*(left@right)/rank**0.5
    result[tensor_name]=weight+update.to(weight.dtype)
    return InterventionResult(state_dict=result,record=InterventionRecord(kind="lora_edit",label=f"{tensor_name}:r{rank}:s{scale:g}",tensor_name=tensor_name,rank=rank,scale=scale,seed=seed,metadata={"update_frobenius_norm":float(update.norm().item())},),changed_tensors=(tensor_name,),changed_entries=weight.numel(),)
def apply_sparse_weight_edit(state_dict:Mapping[str,Tensor],*,tensor_name:str,count:int,scale:float,seed:int,)->InterventionResult:
    result=_clone_state(state_dict)
    if tensor_name not in result:
        raise KeyError(tensor_name)
    tensor=result[tensor_name]
    if count<=0 or count>tensor.numel():
        raise ValueError("count must be between one and tensor.numel()")
    generator=torch.Generator(device=tensor.device).manual_seed(seed)
    indices=torch.randperm(tensor.numel(),generator=generator,device=tensor.device)[:count]
    signs=torch.randint(0,2,(count,),generator=generator,device=tensor.device)*2-1
    flat=tensor.reshape(-1)
    reference_scale=tensor.float().std(unbiased=False).clamp_min(1e-6)
    flat[indices]+=(scale*reference_scale*signs).to(flat.dtype)
    return InterventionResult(state_dict=result,record=InterventionRecord(kind="sparse_weight_edit",label=f"{tensor_name}:k{count}:s{scale:g}",tensor_name=tensor_name,scale=scale,seed=seed,metadata={"count":count,"flat_indices":indices.cpu().tolist()},),changed_tensors=(tensor_name,),changed_entries=count,)
