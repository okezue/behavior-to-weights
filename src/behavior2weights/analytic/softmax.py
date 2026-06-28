from __future__ import annotations
import dataclasses
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from behavior2weights.utils import stable_hash,write_jsonl
@dataclasses.dataclass(frozen=True,slots=True)
class LinearSoftmaxExperimentConfig:
    input_dims:tuple[int,...]=(2,4,8)
    class_counts:tuple[int,...]=(3,5)
    query_counts:tuple[int,...]=(2,4,8,16)
    query_strategies:tuple[str,...]=("gaussian","basis")
    observation_channels:tuple[str,...]=("probabilities","samples")
    sample_counts:tuple[int,...]=(8,32,128)
    systems_per_cell:int=100
    holdout_queries:int=512
    weight_scale:float=1.0
    query_scale:float=1.0
    dirichlet_smoothing:float=0.5
    ridge:float=0.0
    seed:int=20260621
    @classmethod
    def from_dict(cls,raw:Mapping[str,Any])->LinearSoftmaxExperimentConfig:
        known={field.name for field in dataclasses.fields(cls)}
        unknown=set(raw)-known
        if unknown:
            raise ValueError(f"Unknown LinearSoftmaxExperimentConfig fields: {sorted(unknown)}")
        data=dict(raw)
        for name in("input_dims","class_counts","query_counts","sample_counts",):
            if name in data:
                data[name]=tuple(int(value)for value in data[name])
        for name in("query_strategies","observation_channels"):
            if name in data:
                data[name]=tuple(str(value)for value in data[name])
        return cls(**data)
    def __post_init__(self)->None:
        if not self.input_dims or min(self.input_dims)<=0:
            raise ValueError("input_dims must contain positive values")
        if not self.class_counts or min(self.class_counts)<2:
            raise ValueError("class_counts must be at least two")
        if not self.query_counts or min(self.query_counts)<=0:
            raise ValueError("query_counts must contain positive values")
        if set(self.query_strategies)-{"gaussian","basis"}:
            raise ValueError("query_strategies must be gaussian and/or basis")
        if set(self.observation_channels)-{"probabilities","samples"}:
            raise ValueError("observation_channels must be probabilities and/or samples")
        if "samples" in self.observation_channels and(not self.sample_counts or min(self.sample_counts)<=0):
            raise ValueError("sample_counts must be positive for sampled observations")
        if self.systems_per_cell<=0 or self.holdout_queries<=0:
            raise ValueError("systems_per_cell and holdout_queries must be positive")
        if self.weight_scale<=0 or self.query_scale<=0:
            raise ValueError("weight_scale and query_scale must be positive")
        if self.dirichlet_smoothing<=0:
            raise ValueError("dirichlet_smoothing must be positive")
        if self.ridge<0:
            raise ValueError("ridge cannot be negative")
@dataclasses.dataclass(frozen=True,slots=True)
class LinearSoftmaxRecovery:
    recovered_beta:Tensor
    design_rank:int
    design_condition_number:float
    observable_dimension:int
    orbit_dimension:int
def canonical_softmax_parameters(weight:Tensor,bias:Tensor)->Tensor:
    if weight.ndim!=2 or bias.ndim!=1 or weight.shape[0]!=bias.shape[0]:
        raise ValueError("weight/bias shapes must be [classes, input_dim] and [classes]")
    relative_weight=weight[:-1]-weight[-1]
    relative_bias=bias[:-1]-bias[-1]
    return torch.cat([relative_weight,relative_bias[:,None]],dim=1).transpose(0,1)
def probabilities_from_canonical(design:Tensor,beta:Tensor)->Tensor:
    if design.ndim!=2 or beta.ndim!=2 or design.shape[1]!=beta.shape[0]:
        raise ValueError("design and beta dimensions are incompatible")
    relative_logits=design@beta
    reference=torch.zeros(design.shape[0],1,dtype=relative_logits.dtype,device=relative_logits.device)
    return F.softmax(torch.cat([relative_logits,reference],dim=1),dim=1)
def _matrix_rank_and_condition(design:Tensor)->tuple[int,float]:
    singular=torch.linalg.svdvals(design.double())
    if singular.numel()==0:
        return 0,math.inf
    tolerance=max(design.shape)*torch.finfo(singular.dtype).eps*float(singular.max())
    nonzero=singular[singular>tolerance]
    rank=int(nonzero.numel())
    if rank<design.shape[1]or rank==0:
        return rank,math.inf
    return rank,float((nonzero.max()/nonzero.min()).item())
def recover_linear_softmax(design:Tensor,observations:Tensor,*,observation_channel:str,sample_count:int|None=None,smoothing:float=0.5,ridge:float=0.0,)->LinearSoftmaxRecovery:
    if design.ndim!=2 or observations.ndim!=2:
        raise ValueError("design and observations must be matrices")
    if design.shape[0]!=observations.shape[0]or observations.shape[1]<2:
        raise ValueError("observations must align with design rows and contain at least two classes")
    if observation_channel=="probabilities":
        probabilities=observations.double().clamp_min(torch.finfo(torch.float64).tiny)
        probabilities=probabilities/probabilities.sum(dim=1,keepdim=True)
    elif observation_channel=="samples":
        if sample_count is None or sample_count<=0:
            raise ValueError("sample_count must be positive for sampled observations")
        counts=observations.double()
        if bool((counts<0).any()):
            raise ValueError("sample counts cannot be negative")
        probabilities=(counts+smoothing)/(counts.sum(dim=1,keepdim=True)+smoothing*counts.shape[1])
    else:
        raise ValueError("observation_channel must be probabilities or samples")
    log_odds=probabilities[:,:-1].log()-probabilities[:,-1:].log()
    matrix=design.double()
    if ridge>0:
        gram=matrix.transpose(0,1)@matrix
        rhs=matrix.transpose(0,1)@log_odds
        recovered=torch.linalg.solve(gram+ridge*torch.eye(gram.shape[0],dtype=gram.dtype),rhs)
    else:
        recovered=torch.linalg.lstsq(matrix,log_odds).solution
    rank,condition=_matrix_rank_and_condition(matrix)
    classes=observations.shape[1]
    return LinearSoftmaxRecovery(recovered_beta=recovered,design_rank=rank,design_condition_number=condition,observable_dimension=rank*(classes-1),orbit_dimension=design.shape[1]*(classes-1),)
def _design_matrix(query_count:int,input_dim:int,strategy:str,*,query_scale:float,generator:torch.Generator,)->Tensor:
    if strategy=="gaussian":
        inputs=torch.randn(query_count,input_dim,generator=generator)*query_scale
    elif strategy=="basis":
        anchors=torch.cat([torch.zeros(1,input_dim),torch.eye(input_dim)*query_scale],dim=0)
        if query_count<=len(anchors):
            inputs=anchors[:query_count]
        else:
            extra=(torch.randn(query_count-len(anchors),input_dim,generator=generator)*query_scale)
            inputs=torch.cat([anchors,extra],dim=0)
    else:
        raise ValueError(f"unknown query strategy: {strategy}")
    return torch.cat([inputs,torch.ones(query_count,1)],dim=1)
def _system_seed(base_seed:int,payload:Mapping[str,Any])->int:
    return int(stable_hash({"base_seed":base_seed,**dict(payload)},length=16),16)%(2**63-1)
def _normalized_rmse(estimate:Tensor,target:Tensor)->float:
    numerator=(estimate-target).square().mean().sqrt()
    denominator=target.square().mean().sqrt().clamp_min(1e-12)
    return float((numerator/denominator).item())
def _forward_kl(target:Tensor,estimate:Tensor)->float:
    target=target.double().clamp_min(1e-15)
    estimate=estimate.double().clamp_min(1e-15)
    return float((target*(target.log()-estimate.log())).sum(dim=-1).mean().item())
def run_linear_softmax_experiment(config:LinearSoftmaxExperimentConfig,output_directory:str|Path,)->dict[str,Any]:
    output=Path(output_directory)
    output.mkdir(parents=True,exist_ok=True)
    rows:list[dict[str,Any]]=[]
    for input_dim in config.input_dims:
        for class_count in config.class_counts:
            for query_count in config.query_counts:
                for strategy in config.query_strategies:
                    for channel in config.observation_channels:
                        sample_grid:tuple[int|None,...]=(tuple(config.sample_counts)if channel=="samples" else(None,))
                        for sample_count in sample_grid:
                            for replicate in range(config.systems_per_cell):
                                factors={"input_dim":input_dim,"class_count":class_count,"query_count":query_count,"query_strategy":strategy,"observation_channel":channel,"sample_count":sample_count,"replicate":replicate,}
                                seed=_system_seed(config.seed,factors)
                                generator=torch.Generator().manual_seed(seed)
                                weight=(torch.randn(class_count,input_dim,generator=generator)*config.weight_scale)
                                bias=(torch.randn(class_count,generator=generator)*config.weight_scale)
                                beta=canonical_softmax_parameters(weight,bias).double()
                                design=_design_matrix(query_count,input_dim,strategy,query_scale=config.query_scale,generator=generator,).double()
                                probabilities=probabilities_from_canonical(design,beta)
                                if channel=="probabilities":
                                    observations=probabilities
                                else:
                                    assert sample_count is not None
                                    numpy_rng=np.random.default_rng(seed+1)
                                    counts=np.stack([numpy_rng.multinomial(sample_count,row.cpu().numpy())for row in probabilities])
                                    observations=torch.from_numpy(counts)
                                recovery=recover_linear_softmax(design,observations,observation_channel=channel,sample_count=sample_count,smoothing=config.dirichlet_smoothing,ridge=config.ridge,)
                                holdout=torch.cat([torch.randn(config.holdout_queries,input_dim,generator=generator,)*config.query_scale,torch.ones(config.holdout_queries,1),],dim=1,).double()
                                true_probabilities=probabilities_from_canonical(holdout,beta)
                                recovered_probabilities=probabilities_from_canonical(holdout,recovery.recovered_beta)
                                rank_fraction=(recovery.observable_dimension/recovery.orbit_dimension)
                                rows.append({"system_id":f"softmax-{stable_hash(factors, length=24)}",**factors,"seed":seed,"design_rank":recovery.design_rank,"design_columns":input_dim+1,"design_condition_number":(recovery.design_condition_number if math.isfinite(recovery.design_condition_number)else None),"observable_dimension":recovery.observable_dimension,"orbit_dimension":recovery.orbit_dimension,"observable_fraction":rank_fraction,"orbit_nrmse":_normalized_rmse(recovery.recovered_beta,beta),"functional_forward_kl":_forward_kl(true_probabilities,recovered_probabilities,),"full_rank":recovery.design_rank==input_dim+1,"exact_recovery":(channel=="probabilities" and recovery.design_rank==input_dim+1 and _normalized_rmse(recovery.recovered_beta,beta)<1e-8),})
    result_path=output/"results.jsonl"
    write_jsonl(result_path,rows)
    grouped:dict[tuple[Any,...],list[dict[str,Any]]]=defaultdict(list)
    keys=("input_dim","class_count","query_count","query_strategy","observation_channel","sample_count",)
    for row in rows:
        grouped[tuple(row[key]for key in keys)].append(row)
    cell_summaries:list[dict[str,Any]]=[]
    for cell,values in grouped.items():
        cell_summaries.append({**dict(zip(keys,cell,strict=True)),"systems":len(values),"full_rank_rate":float(np.mean([row["full_rank"]for row in values])),"exact_recovery_rate":float(np.mean([row["exact_recovery"]for row in values])),"mean_orbit_nrmse":float(np.mean([row["orbit_nrmse"]for row in values])),"mean_functional_forward_kl":float(np.mean([row["functional_forward_kl"]for row in values])),"mean_observable_fraction":float(np.mean([row["observable_fraction"]for row in values])),})
    summary={"schema_version":1,"config":dataclasses.asdict(config),"systems":len(rows),"cells":len(cell_summaries),"result_path":str(result_path),"cell_summaries":cell_summaries,"interpretation":("Full-probability recovery is exact only when the augmented query design has full " "column rank; sampled outputs contract toward the same orbit as repetitions grow."),}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary
