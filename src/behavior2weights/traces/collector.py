from __future__ import annotations
import dataclasses
from collections.abc import Callable,Sequence
from pathlib import Path
from typing import Any
import torch
from torch import nn
from behavior2weights.schemas import ObservationChannel,QueryRecord,TargetRecord
from behavior2weights.traces.observations import ObservationConfig,collect_observations
from behavior2weights.traces.store import TraceBundle,save_trace_bundle
from behavior2weights.utils import stable_hash
@dataclasses.dataclass(frozen=True,slots=True)
class CollectionConfig:
    batch_size:int=128
    device:str="cpu"
    base_seed:int=0
    fail_fast:bool=True
ModelLoader=Callable[[TargetRecord],nn.Module]
def _sampling_seed(*,base_seed:int,target_id:str,query_id:str,channel:ObservationChannel,)->int:
    payload={"scheme":"behavior2weights-trace-rng-v1","base_seed":int(base_seed),"target_id":target_id,"query_id":query_id,"channel":channel.value,}
    return int(stable_hash(payload,length=16),16)%(2**63-1)
def collect_target_traces(targets:Sequence[TargetRecord],queries:Sequence[QueryRecord],model_loader:ModelLoader,observation_config:ObservationConfig,collection_config:CollectionConfig,*,output_directory:str|Path|None=None,)->TraceBundle:
    if not targets:
        raise ValueError("targets cannot be empty")
    if not queries:
        raise ValueError("queries cannot be empty")
    sequence_lengths={len(query.input_ids)for query in queries}
    if len(sequence_lengths)!=1:
        raise ValueError("all queries must have equal sequence length for tensor storage")
    input_ids=torch.tensor([query.input_ids for query in queries],dtype=torch.long)
    target_ids:list[str]=[]
    trace_tensors:list[torch.Tensor]=[]
    auxiliary_tensors:dict[str,list[torch.Tensor]]={}
    failures:list[dict[str,Any]]=[]
    stochastic=observation_config.channel in{ObservationChannel.TOKENS,ObservationChannel.SAMPLE_HISTOGRAM,}
    for target in targets:
        try:
            model=model_loader(target)
            row_seeds=([_sampling_seed(base_seed=collection_config.base_seed,target_id=target.target_id,query_id=query.query_id,channel=observation_config.channel,)for query in queries]if stochastic else None)
            observations,auxiliary=collect_observations(model,input_ids,observation_config,batch_size=collection_config.batch_size,device=collection_config.device,seed=collection_config.base_seed,row_seeds=row_seeds,)
            target_ids.append(target.target_id)
            trace_tensors.append(observations)
            for key,value in auxiliary.items():
                auxiliary_tensors.setdefault(key,[]).append(value)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:
            failures.append({"target_id":target.target_id,"error":repr(error)})
            if collection_config.fail_fast:
                raise
    if not trace_tensors:
        raise RuntimeError("no traces were collected")
    bundle=TraceBundle(target_ids=tuple(target_ids),query_ids=tuple(query.query_id for query in queries),input_ids=input_ids,observations=torch.stack(trace_tensors),channel=observation_config.channel,feature_dim=observation_config.feature_dim,auxiliary={key:torch.stack(values)for key,values in auxiliary_tensors.items()},metadata={"observation_config":dataclasses.asdict(observation_config),"collection_config":dataclasses.asdict(collection_config),"failures":failures,"rng_scheme":("sha256(base_seed,target_id,query_id,channel)-v1" if stochastic else "deterministic-channel"),},)
    if output_directory is not None:
        save_trace_bundle(bundle,output_directory)
    return bundle
