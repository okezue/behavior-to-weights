from __future__ import annotations
import dataclasses
from collections.abc import Mapping
from typing import Any,TypedDict,cast
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from behavior2weights.models.weight_space import AddressSpace
from behavior2weights.schemas import ObservationChannel
CHANNEL_TO_ID={channel:index for index,channel in enumerate(ObservationChannel)}
class BehaviorToWeightsOutput(TypedDict):
    latent:Tensor
    weight_mean:Tensor
    weight_log_variance:Tensor
    properties:dict[str,Tensor]
@dataclasses.dataclass(frozen=True,slots=True)
class InverseModelConfig:
    vocab_size:int=32
    max_seq_len:int=16
    observation_dim:int=32
    trace_width:int=64
    trace_heads:int=4
    query_layers:int=1
    set_layers:int=2
    latent_dim:int=64
    address_width:int=64
    decoder_width:int=128
    decoder_layers:int=3
    max_tensors:int=256
    max_layers:int=64
    role_count:int=32
    dropout:float=0.0
    min_log_variance:float=-10.0
    max_log_variance:float=4.0
    property_dims:dict[str,int]=dataclasses.field(default_factory=dict)
    def __post_init__(self)->None:
        if self.trace_width%self.trace_heads!=0:
            raise ValueError("trace_width must be divisible by trace_heads")
        if self.max_seq_len<=0 or self.vocab_size<=0 or self.observation_dim<=0:
            raise ValueError("vocab_size, max_seq_len, and observation_dim must be positive")
        if self.decoder_layers<1:
            raise ValueError("decoder_layers must be at least one")
    @classmethod
    def from_dict(cls,data:Mapping[str,Any])->InverseModelConfig:
        fields={field.name for field in dataclasses.fields(cls)}
        unknown=set(data)-fields
        if unknown:
            raise ValueError(f"Unknown InverseModelConfig fields: {sorted(unknown)}")
        return cls(**dict(data))
class QueryEncoder(nn.Module):
    def __init__(self,config:InverseModelConfig)->None:
        super().__init__()
        self.token_embedding=nn.Embedding(config.vocab_size,config.trace_width)
        self.position_embedding=nn.Embedding(config.max_seq_len,config.trace_width)
        layer=nn.TransformerEncoderLayer(d_model=config.trace_width,nhead=config.trace_heads,dim_feedforward=4*config.trace_width,dropout=config.dropout,activation="gelu",batch_first=True,norm_first=True,)
        self.transformer=nn.TransformerEncoder(layer,num_layers=config.query_layers)
        self.norm=nn.LayerNorm(config.trace_width)
    def forward(self,input_ids:Tensor,token_mask:Tensor|None=None)->Tensor:
        if input_ids.ndim!=3:
            raise ValueError("input_ids must have shape [batch, queries, sequence]")
        batch,queries,sequence=input_ids.shape
        if sequence>self.position_embedding.num_embeddings:
            raise ValueError("query length exceeds configured max_seq_len")
        flattened=input_ids.reshape(batch*queries,sequence)
        positions=torch.arange(sequence,device=input_ids.device)
        encoded=self.token_embedding(flattened)+self.position_embedding(positions)[None,:,:]
        flat_mask:Tensor|None=None
        if token_mask is not None:
            flat_mask=~token_mask.reshape(batch*queries,sequence).bool()
        encoded=self.transformer(encoded,src_key_padding_mask=flat_mask)
        if token_mask is None:
            pooled=encoded.mean(dim=1)
        else:
            weights=token_mask.reshape(batch*queries,sequence).to(encoded.dtype)
            pooled=(encoded*weights[...,None]).sum(dim=1)/weights.sum(dim=1,keepdim=True).clamp_min(1)
        return cast(Tensor,self.norm(pooled).reshape(batch,queries,-1))
class TraceSetEncoder(nn.Module):
    def __init__(self,config:InverseModelConfig)->None:
        super().__init__()
        self.query_encoder=QueryEncoder(config)
        self.observation_projection=nn.Sequential(nn.LayerNorm(config.observation_dim),nn.Linear(config.observation_dim,config.trace_width),nn.GELU(),nn.Linear(config.trace_width,config.trace_width),)
        self.channel_embedding=nn.Embedding(len(CHANNEL_TO_ID),config.trace_width)
        self.pair_projection=nn.Sequential(nn.LayerNorm(3*config.trace_width),nn.Linear(3*config.trace_width,config.trace_width),nn.GELU(),)
        self.summary=nn.Parameter(torch.zeros(1,1,config.trace_width))
        layer=nn.TransformerEncoderLayer(d_model=config.trace_width,nhead=config.trace_heads,dim_feedforward=4*config.trace_width,dropout=config.dropout,activation="gelu",batch_first=True,norm_first=True,)
        self.set_transformer=nn.TransformerEncoder(layer,num_layers=config.set_layers)
        self.to_latent=nn.Sequential(nn.LayerNorm(config.trace_width),nn.Linear(config.trace_width,config.latent_dim),)
        nn.init.normal_(self.summary,std=0.02)
    def forward(self,input_ids:Tensor,observations:Tensor,channel_ids:Tensor,*,query_mask:Tensor|None=None,token_mask:Tensor|None=None,)->Tensor:
        if observations.ndim!=3:
            raise ValueError("observations must have shape [batch, queries, observation_dim]")
        if input_ids.shape[:2]!=observations.shape[:2]:
            raise ValueError("input_ids and observations must share batch and query dimensions")
        if channel_ids.shape!=input_ids.shape[:2]:
            raise ValueError("channel_ids must have shape [batch, queries]")
        query_state=self.query_encoder(input_ids,token_mask)
        observation_state=self.observation_projection(observations)
        channel_state=self.channel_embedding(channel_ids)
        pair_state=self.pair_projection(torch.cat([query_state,observation_state,channel_state],dim=-1))
        summary=self.summary.expand(input_ids.shape[0],-1,-1)
        sequence=torch.cat([summary,pair_state],dim=1)
        padding_mask:Tensor|None=None
        if query_mask is not None:
            summary_valid=torch.ones(query_mask.shape[0],1,dtype=torch.bool,device=query_mask.device)
            padding_mask=~torch.cat([summary_valid,query_mask.bool()],dim=1)
        encoded=self.set_transformer(sequence,src_key_padding_mask=padding_mask)
        return cast(Tensor,self.to_latent(encoded[:,0]))
class CoordinateDecoder(nn.Module):
    def __init__(self,config:InverseModelConfig)->None:
        super().__init__()
        width=config.address_width
        self.tensor_embedding=nn.Embedding(config.max_tensors,width)
        self.role_embedding=nn.Embedding(config.role_count,width)
        self.layer_embedding=nn.Embedding(config.max_layers+1,width)
        self.continuous_projection=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width),)
        self.address_projection=nn.Sequential(nn.LayerNorm(4*width),nn.Linear(4*width,width),nn.GELU(),)
        layers:list[nn.Module]=[]
        input_width=config.latent_dim+width
        for _layer_index in range(config.decoder_layers):
            output_width=config.decoder_width
            layers.extend([nn.Linear(input_width,output_width),nn.GELU(),nn.Dropout(config.dropout)])
            input_width=output_width
        self.body=nn.Sequential(*layers)
        self.mean_head=nn.Linear(input_width,1)
        self.log_variance_head=nn.Linear(input_width,1)
        self.min_log_variance=config.min_log_variance
        self.max_log_variance=config.max_log_variance
    def forward(self,latent:Tensor,descriptors:Mapping[str,Tensor])->tuple[Tensor,Tensor]:
        tensor_id=descriptors["tensor_id"].to(latent.device)
        role_id=descriptors["role_id"].to(latent.device)
        layer_id=descriptors["layer_id"].to(latent.device)
        continuous=descriptors["continuous"].to(latent.device,latent.dtype)
        if tensor_id.ndim==1:
            tensor_id=tensor_id.unsqueeze(0).expand(latent.shape[0],-1)
            role_id=role_id.unsqueeze(0).expand(latent.shape[0],-1)
            layer_id=layer_id.unsqueeze(0).expand(latent.shape[0],-1)
            continuous=continuous.unsqueeze(0).expand(latent.shape[0],-1,-1)
        if tensor_id.shape[0]!=latent.shape[0]:
            raise ValueError("descriptor batch dimension must match latent batch")
        address=self.address_projection(torch.cat([self.tensor_embedding(tensor_id),self.role_embedding(role_id),self.layer_embedding(layer_id),self.continuous_projection(continuous),],dim=-1,))
        expanded_latent=latent[:,None,:].expand(-1,tensor_id.shape[1],-1)
        hidden=self.body(torch.cat([expanded_latent,address],dim=-1))
        mean=self.mean_head(hidden).squeeze(-1)
        log_variance=(self.log_variance_head(hidden).squeeze(-1).clamp(self.min_log_variance,self.max_log_variance))
        return mean,log_variance
class BehaviorToWeights(nn.Module):
    def __init__(self,config:InverseModelConfig)->None:
        super().__init__()
        self.config=config
        self.trace_encoder=TraceSetEncoder(config)
        self.weight_decoder=CoordinateDecoder(config)
        self.property_heads=nn.ModuleDict({name:nn.Linear(config.latent_dim,count)for name,count in config.property_dims.items()})
    def encode(self,input_ids:Tensor,observations:Tensor,channel_ids:Tensor,*,query_mask:Tensor|None=None,token_mask:Tensor|None=None,)->Tensor:
        return cast(Tensor,self.trace_encoder(input_ids,observations,channel_ids,query_mask=query_mask,token_mask=token_mask,),)
    def forward(self,input_ids:Tensor,observations:Tensor,channel_ids:Tensor,descriptors:Mapping[str,Tensor],*,query_mask:Tensor|None=None,token_mask:Tensor|None=None,)->BehaviorToWeightsOutput:
        latent=self.encode(input_ids,observations,channel_ids,query_mask=query_mask,token_mask=token_mask,)
        mean,log_variance=self.weight_decoder(latent,descriptors)
        properties={name:head(latent)for name,head in self.property_heads.items()}
        return{"latent":latent,"weight_mean":mean,"weight_log_variance":log_variance,"properties":properties,}
    @torch.no_grad()
    def decode_all(self,latent:Tensor,address_space:AddressSpace,*,chunk_size:int=65_536,)->tuple[Tensor,Tensor]:
        means:list[Tensor]=[]
        log_variances:list[Tensor]=[]
        for start in range(0,address_space.total_parameters,chunk_size):
            end=min(start+chunk_size,address_space.total_parameters)
            descriptors=address_space.descriptors(torch.arange(start,end))
            mean,log_variance=self.weight_decoder(latent,descriptors)
            means.append(mean.cpu())
            log_variances.append(log_variance.cpu())
        return torch.cat(means,dim=1),torch.cat(log_variances,dim=1)
def gaussian_nll(target:Tensor,mean:Tensor,log_variance:Tensor,*,reduction:str="mean",)->Tensor:
    loss=0.5*(log_variance+(target-mean).square()*torch.exp(-log_variance))
    if reduction=="none":
        return loss
    if reduction=="sum":
        return loss.sum()
    if reduction=="mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
def posterior_sample(mean:Tensor,log_variance:Tensor,generator:torch.Generator|None=None)->Tensor:
    noise=torch.randn(mean.shape,dtype=mean.dtype,device=mean.device,generator=generator)
    return mean+torch.exp(0.5*log_variance)*noise
def property_loss(logits:Mapping[str,Tensor],labels:Mapping[str,Tensor],*,ignore_index:int=-100,)->Tensor:
    losses:list[Tensor]=[]
    for name,output in logits.items():
        if name not in labels:
            continue
        losses.append(F.cross_entropy(output,labels[name].to(output.device),ignore_index=ignore_index))
    if not losses:
        return torch.zeros((),device=next(iter(logits.values())).device if logits else "cpu")
    return torch.stack(losses).mean()
