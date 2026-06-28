from __future__ import annotations
import dataclasses
from collections.abc import Callable,Mapping,Sequence
from typing import cast
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from behavior2weights.schemas import ObservationChannel
@dataclasses.dataclass(frozen=True,slots=True)
class ObservationConfig:
    channel:ObservationChannel=ObservationChannel.LOGITS
    vocab_size:int=32
    feature_dim:int=32
    topk:int=8
    sample_count:int=32
    temperature:float=1.0
    sketch_dim:int=16
    sketch_seed:int=17
    center_logits:bool=True
    @classmethod
    def from_dict(cls,raw:Mapping[str,object],*,vocab_size:int,)->ObservationConfig:
        data=dict(raw)
        channel_value=data.pop("name",data.pop("channel",ObservationChannel.LOGITS.value))
        channel=ObservationChannel(str(channel_value))
        fields=dataclasses.fields(cls)
        defaults={field.name:field.default for field in fields}
        known={field.name for field in fields}-{"channel","vocab_size"}
        unknown=set(data)-known
        if unknown:
            raise ValueError(f"Unknown ObservationConfig fields: {sorted(unknown)}")
        if "feature_dim" not in data:
            if channel==ObservationChannel.LOGIT_SKETCH:
                data["feature_dim"]=int(cast(int,data.get("sketch_dim",defaults["sketch_dim"])))
            elif channel==ObservationChannel.TOPK:
                data["feature_dim"]=2*int(cast(int,data.get("topk",defaults["topk"])))
            elif channel in{ObservationChannel.TOKENS,ObservationChannel.SAMPLE_HISTOGRAM}:
                data["feature_dim"]=min(vocab_size,512)
            else:
                data["feature_dim"]=vocab_size
        return cls(channel=channel,vocab_size=vocab_size,**data)
    def __post_init__(self)->None:
        if self.vocab_size<=0 or self.feature_dim<=0:
            raise ValueError("vocab_size and feature_dim must be positive")
        if not 1<=self.topk<=self.vocab_size:
            raise ValueError("topk must be between one and vocab_size")
        if self.sample_count<=0:
            raise ValueError("sample_count must be positive")
        if self.temperature<=0:
            raise ValueError("temperature must be positive")
        if self.channel==ObservationChannel.LOGIT_SKETCH and self.sketch_dim>self.feature_dim:
            raise ValueError("sketch_dim cannot exceed feature_dim")
        if self.channel==ObservationChannel.TOPK and 2*self.topk>self.feature_dim:
            raise ValueError("top-k observations require feature_dim >= 2 * topk")
        if self.channel==ObservationChannel.LOGITS and self.vocab_size>self.feature_dim:
            raise ValueError("full-logit observations require feature_dim >= vocab_size; use logit_sketch " "for dimensionality reduction")
def _pad_or_truncate(values:Tensor,width:int)->Tensor:
    if values.shape[-1]==width:
        return values
    if values.shape[-1]>width:
        return values[...,:width]
    return F.pad(values,(0,width-values.shape[-1]))
def sketch_matrix(config:ObservationConfig,*,device:torch.device,dtype:torch.dtype)->Tensor:
    generator=torch.Generator(device=device).manual_seed(config.sketch_seed)
    matrix=torch.randn(config.vocab_size,config.sketch_dim,generator=generator,device=device,dtype=dtype,)
    return cast(Tensor,matrix/config.sketch_dim**0.5)
def encode_logits(logits:Tensor,config:ObservationConfig,*,generator:torch.Generator|None=None,)->tuple[Tensor,dict[str,Tensor]]:
    if logits.ndim!=2 or logits.shape[-1]!=config.vocab_size:
        raise ValueError(f"logits must have shape [batch, {config.vocab_size}], got {tuple(logits.shape)}")
    scaled=logits.float()/config.temperature
    centered=scaled-scaled.mean(dim=-1,keepdim=True)if config.center_logits else scaled
    probabilities=F.softmax(scaled,dim=-1)
    metadata:dict[str,Tensor]={}
    if config.channel==ObservationChannel.LOGITS:
        observation=centered
    elif config.channel==ObservationChannel.TOPK:
        values,indices=torch.topk(F.log_softmax(scaled,dim=-1),k=config.topk,dim=-1)
        denominator=max(config.vocab_size-1,1)
        normalized_indices=indices.float()/denominator
        observation=torch.cat([normalized_indices,values],dim=-1)
        metadata["topk_indices"]=indices
        metadata["topk_values"]=values
    elif config.channel==ObservationChannel.TOKENS:
        sampled=torch.multinomial(probabilities,1,generator=generator).squeeze(-1)
        buckets=sampled.remainder(config.feature_dim)
        observation=F.one_hot(buckets,num_classes=config.feature_dim).float()
        metadata["output_ids"]=sampled[:,None]
    elif config.channel==ObservationChannel.SAMPLE_HISTOGRAM:
        sampled=torch.multinomial(probabilities,config.sample_count,replacement=True,generator=generator,)
        buckets=sampled.remainder(config.feature_dim)
        observation=torch.zeros(probabilities.shape[0],config.feature_dim,dtype=probabilities.dtype,device=probabilities.device,)
        observation.scatter_add_(1,buckets,torch.ones_like(buckets,dtype=observation.dtype),)
        observation/=config.sample_count
        metadata["output_ids"]=sampled
    elif config.channel==ObservationChannel.LOGIT_SKETCH:
        observation=centered@sketch_matrix(config,device=centered.device,dtype=centered.dtype)
    else:
        raise ValueError(f"Unsupported observation channel: {config.channel}")
    return _pad_or_truncate(observation,config.feature_dim),metadata
@torch.no_grad()
def collect_observations(model:nn.Module,input_ids:Tensor,config:ObservationConfig,*,batch_size:int=128,device:str|torch.device="cpu",seed:int=0,row_seeds:Sequence[int]|Tensor|None=None,logits_selector:Callable[[Tensor],Tensor]|None=None,)->tuple[Tensor,dict[str,Tensor]]:
    if input_ids.ndim!=2:
        raise ValueError("input_ids must have shape [queries, sequence]")
    model=model.to(device)
    model.eval()
    generator_device=torch.device(device)
    generator=torch.Generator(device=generator_device).manual_seed(seed)
    if row_seeds is not None:
        row_seeds=[int(value)for value in row_seeds]
        if len(row_seeds)!=input_ids.shape[0]:
            raise ValueError("row_seeds must have one seed per query")
    observations:list[Tensor]=[]
    metadata_accumulator:dict[str,list[Tensor]]={}
    for start in range(0,input_ids.shape[0],batch_size):
        batch=input_ids[start:start+batch_size].to(device)
        output=model(batch)
        if isinstance(output,tuple):
            output=output[0]
        if hasattr(output,"logits"):
            output=output.logits
        if not isinstance(output,Tensor):
            raise TypeError("model output must be a Tensor, tuple containing a Tensor, or have .logits")
        logits=logits_selector(output)if logits_selector else output[:,-1,:]
        if row_seeds is None:
            encoded,metadata=encode_logits(logits,config,generator=generator)
        else:
            encoded_rows:list[Tensor]=[]
            metadata_rows:dict[str,list[Tensor]]={}
            for local_index in range(logits.shape[0]):
                row_generator=torch.Generator(device=generator_device).manual_seed(row_seeds[start+local_index])
                row_encoded,row_metadata=encode_logits(logits[local_index:local_index+1],config,generator=row_generator,)
                encoded_rows.append(row_encoded)
                for key,value in row_metadata.items():
                    metadata_rows.setdefault(key,[]).append(value)
            encoded=torch.cat(encoded_rows,dim=0)
            metadata={key:torch.cat(values,dim=0)for key,values in metadata_rows.items()}
        observations.append(encoded.cpu())
        for key,value in metadata.items():
            metadata_accumulator.setdefault(key,[]).append(value.cpu())
    return torch.cat(observations),{key:torch.cat(values)for key,values in metadata_accumulator.items()}
