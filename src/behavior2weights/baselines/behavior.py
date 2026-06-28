from __future__ import annotations
from typing import cast
import torch
from torch import Tensor
def mean_checkpoint(vectors:Tensor)->Tensor:
    if vectors.ndim!=2:
        raise ValueError("vectors must have shape [models, parameters]")
    return vectors.mean(dim=0)
def nearest_behavior_neighbor(train_observations:Tensor,query_observation:Tensor,*,metric:str="cosine",)->int:
    if train_observations.ndim<2:
        raise ValueError("train_observations must begin with a model dimension")
    train=train_observations.reshape(train_observations.shape[0],-1).float()
    query=query_observation.reshape(1,-1).float()
    if metric=="euclidean":
        distances=torch.cdist(query,train).squeeze(0)
        return int(distances.argmin().item())
    if metric=="cosine":
        train=train/train.norm(dim=1,keepdim=True).clamp_min(1e-12)
        query=query/query.norm(dim=1,keepdim=True).clamp_min(1e-12)
        similarities=query@train.T
        return int(similarities.argmax().item())
    raise ValueError(f"Unknown metric: {metric}")
def ridge_trace_to_weights(train_observations:Tensor,train_weights:Tensor,test_observations:Tensor,*,regularization:float=1e-3,)->Tensor:
    train_x=train_observations.reshape(train_observations.shape[0],-1).float()
    test_x=test_observations.reshape(test_observations.shape[0],-1).float()
    train_y=train_weights.float()
    train_x=torch.cat([train_x,torch.ones(len(train_x),1)],dim=1)
    test_x=torch.cat([test_x,torch.ones(len(test_x),1)],dim=1)
    gram=train_x.T@train_x
    identity=torch.eye(gram.shape[0],dtype=gram.dtype,device=gram.device)
    solution=torch.linalg.solve(gram+regularization*identity,train_x.T@train_y)
    return cast(Tensor,test_x@solution)
