from __future__ import annotations
import dataclasses
from collections.abc import Mapping
from typing import Any
from torch import Tensor,nn
from behavior2weights.models.inverse import InverseModelConfig,TraceSetEncoder
@dataclasses.dataclass(frozen=True,slots=True)
class PropertyModelConfig:
    encoder:InverseModelConfig
    property_dims:dict[str,int]
    @classmethod
    def from_dict(cls,raw:Mapping[str,Any])->PropertyModelConfig:
        data=dict(raw)
        dimensions={str(key):int(value)for key,value in data.pop("property_dims").items()}
        encoder=InverseModelConfig.from_dict({**data,"property_dims":{}})
        return cls(encoder=encoder,property_dims=dimensions)
    def to_dict(self)->dict[str,Any]:
        return{**dataclasses.asdict(self.encoder),"property_dims":self.property_dims}
class BehaviorPropertyClassifier(nn.Module):
    def __init__(self,config:PropertyModelConfig)->None:
        super().__init__()
        self.config=config
        self.encoder=TraceSetEncoder(config.encoder)
        self.heads=nn.ModuleDict({name:nn.Linear(config.encoder.latent_dim,size)for name,size in config.property_dims.items()})
    def forward(self,input_ids:Tensor,observations:Tensor,channel_ids:Tensor,*,query_mask:Tensor|None=None,token_mask:Tensor|None=None,)->dict[str,Tensor]:
        latent=self.encoder(input_ids,observations,channel_ids,query_mask=query_mask,token_mask=token_mask,)
        return{name:head(latent)for name,head in self.heads.items()}
