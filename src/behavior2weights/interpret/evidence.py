from __future__ import annotations
from collections.abc import Mapping
from typing import cast
import torch
from torch import Tensor
from behavior2weights.models.inverse import BehaviorToWeights
def behavior_to_weight_jacobian(model:BehaviorToWeights,input_ids:Tensor,observations:Tensor,channel_ids:Tensor,descriptors:Mapping[str,Tensor],*,query_mask:Tensor|None=None,)->Tensor:
    if input_ids.shape[0]!=1:
        raise ValueError("full Jacobian helper currently supports batch size one")
    observations=observations.detach().requires_grad_(True)
    def prediction(value:Tensor)->Tensor:
        output=model(input_ids,value,channel_ids,descriptors,query_mask=query_mask,)
        return cast(Tensor,output["weight_mean"].squeeze(0))
    return cast(Tensor,torch.autograd.functional.jacobian(prediction,observations,vectorize=True).squeeze(1),)
@torch.no_grad()
def transcript_patch_effect(model:BehaviorToWeights,input_ids_a:Tensor,observations_a:Tensor,input_ids_b:Tensor,observations_b:Tensor,channel_ids:Tensor,descriptors:Mapping[str,Tensor],*,query_indices:Tensor,)->Tensor:
    base=model(input_ids_a,observations_a,channel_ids,descriptors)["weight_mean"]
    patched_inputs=input_ids_a.clone()
    patched_observations=observations_a.clone()
    patched_inputs[:,query_indices]=input_ids_b[:,query_indices]
    patched_observations[:,query_indices]=observations_b[:,query_indices]
    patched=model(patched_inputs,patched_observations,channel_ids,descriptors)["weight_mean"]
    return cast(Tensor,patched-base)
def integrated_gradients_observations(model:BehaviorToWeights,input_ids:Tensor,observations:Tensor,channel_ids:Tensor,descriptors:Mapping[str,Tensor],*,target_coordinate:int,baseline:Tensor|None=None,steps:int=32,)->Tensor:
    baseline=torch.zeros_like(observations)if baseline is None else baseline
    total_gradient=torch.zeros_like(observations)
    for alpha in torch.linspace(0,1,steps,device=observations.device):
        interpolated=(baseline+alpha*(observations-baseline)).detach().requires_grad_(True)
        prediction=model(input_ids,interpolated,channel_ids,descriptors)["weight_mean"]
        scalar=prediction[:,target_coordinate].sum()
        gradient=torch.autograd.grad(scalar,interpolated)[0]
        total_gradient+=gradient
    return cast(Tensor,(observations-baseline)*total_gradient/steps)
