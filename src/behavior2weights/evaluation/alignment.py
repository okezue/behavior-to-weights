from __future__ import annotations
from collections import OrderedDict
from collections.abc import Mapping
from typing import cast
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from behavior2weights.models.micro_transformer import(MicroTransformerConfig,permute_attention_heads,permute_mlp_neurons,)
def _normalized_rows(matrix:Tensor)->Tensor:
    flat=matrix.float().reshape(matrix.shape[0],-1)
    return cast(Tensor,flat/flat.norm(dim=1,keepdim=True).clamp_min(1e-12))
def attention_head_features(state_dict:Mapping[str,Tensor],config:MicroTransformerConfig,layer:int)->Tensor:
    prefix=f"blocks.{layer}.attn"
    features:list[Tensor]=[]
    for head in range(config.n_heads):
        start=head*config.head_dim
        end=(head+1)*config.head_dim
        parts=[state_dict[f"{prefix}.q_proj.weight"][start:end,:],state_dict[f"{prefix}.k_proj.weight"][start:end,:],state_dict[f"{prefix}.v_proj.weight"][start:end,:],state_dict[f"{prefix}.o_proj.weight"][:,start:end],]
        for projection in("q_proj","k_proj","v_proj"):
            bias_name=f"{prefix}.{projection}.bias"
            if bias_name in state_dict:
                parts.append(state_dict[bias_name][start:end])
        features.append(torch.cat([part.reshape(-1).float()for part in parts]))
    return torch.stack(features)
def mlp_neuron_features(state_dict:Mapping[str,Tensor],layer:int)->Tensor:
    prefix=f"blocks.{layer}.mlp"
    fc1=state_dict[f"{prefix}.fc1.weight"]
    fc2=state_dict[f"{prefix}.fc2.weight"]
    bias=state_dict.get(f"{prefix}.fc1.bias")
    features:list[Tensor]=[]
    for neuron in range(fc1.shape[0]):
        parts=[fc1[neuron],fc2[:,neuron]]
        if bias is not None:
            parts.append(bias[neuron].reshape(1))
        features.append(torch.cat([part.reshape(-1).float()for part in parts]))
    return torch.stack(features)
def _assignment(reference:Tensor,candidate:Tensor)->Tensor:
    reference_normalized=_normalized_rows(reference)
    candidate_normalized=_normalized_rows(candidate)
    cosine_cost=1.0-reference_normalized@candidate_normalized.T
    norm_cost=(reference.float().norm(dim=1)[:,None]-candidate.float().norm(dim=1)[None,:]).abs()/reference.float().norm(dim=1).median().clamp_min(1e-8)
    cost=cosine_cost+0.05*norm_cost
    row,column=linear_sum_assignment(cost.detach().cpu().numpy())
    if not np.array_equal(row,np.arange(len(row))):
        column=column[np.argsort(row)]
    return torch.tensor(column,dtype=torch.long)
def align_micro_state_dict(candidate:Mapping[str,Tensor],reference:Mapping[str,Tensor],config:MicroTransformerConfig,)->tuple[OrderedDict[str,Tensor],dict[str,list[list[int]]|list[int]]]:
    aligned=OrderedDict((name,value.detach().clone())for name,value in candidate.items())
    permutations:dict[str,list[list[int]]|list[int]]={"attention":[],"mlp":[]}
    attention_permutations:list[list[int]]=[]
    mlp_permutations:list[list[int]]=[]
    for layer in range(config.n_layers):
        head_permutation=_assignment(attention_head_features(reference,config,layer),attention_head_features(aligned,config,layer),)
        aligned=permute_attention_heads(aligned,config,layer,head_permutation)
        attention_permutations.append(head_permutation.tolist())
        neuron_permutation=_assignment(mlp_neuron_features(reference,layer),mlp_neuron_features(aligned,layer),)
        aligned=permute_mlp_neurons(aligned,layer,neuron_permutation)
        mlp_permutations.append(neuron_permutation.tolist())
    permutations["attention"]=attention_permutations
    permutations["mlp"]=mlp_permutations
    return aligned,permutations
