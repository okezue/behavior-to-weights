from __future__ import annotations
import dataclasses
import json
import math
import re
from collections import OrderedDict
from collections.abc import Mapping,Sequence
from enum import IntEnum
from pathlib import Path
import torch
from torch import Tensor
class ParameterRole(IntEnum):
    OTHER=0
    TOKEN_EMBEDDING=1
    POSITION_EMBEDDING=2
    Q_PROJ_WEIGHT=3
    Q_PROJ_BIAS=4
    K_PROJ_WEIGHT=5
    K_PROJ_BIAS=6
    V_PROJ_WEIGHT=7
    V_PROJ_BIAS=8
    O_PROJ_WEIGHT=9
    O_PROJ_BIAS=10
    MLP_IN_WEIGHT=11
    MLP_IN_BIAS=12
    MLP_OUT_WEIGHT=13
    MLP_OUT_BIAS=14
    NORM_WEIGHT=15
    NORM_BIAS=16
    LM_HEAD=17
ROLE_PATTERNS:tuple[tuple[str,ParameterRole],...]=(("token_embedding.weight",ParameterRole.TOKEN_EMBEDDING),("position_embedding.weight",ParameterRole.POSITION_EMBEDDING),("q_proj.weight",ParameterRole.Q_PROJ_WEIGHT),("q_proj.bias",ParameterRole.Q_PROJ_BIAS),("k_proj.weight",ParameterRole.K_PROJ_WEIGHT),("k_proj.bias",ParameterRole.K_PROJ_BIAS),("v_proj.weight",ParameterRole.V_PROJ_WEIGHT),("v_proj.bias",ParameterRole.V_PROJ_BIAS),("o_proj.weight",ParameterRole.O_PROJ_WEIGHT),("o_proj.bias",ParameterRole.O_PROJ_BIAS),("mlp.fc1.weight",ParameterRole.MLP_IN_WEIGHT),("mlp.fc1.bias",ParameterRole.MLP_IN_BIAS),("mlp.fc2.weight",ParameterRole.MLP_OUT_WEIGHT),("mlp.fc2.bias",ParameterRole.MLP_OUT_BIAS),("lm_head.weight",ParameterRole.LM_HEAD),)
@dataclasses.dataclass(frozen=True,slots=True)
class TensorSpec:
    name:str
    shape:tuple[int,...]
    role:ParameterRole
    layer:int
    tensor_id:int
    offset:int
    length:int
    @property
    def rank(self)->int:
        return len(self.shape)
class AddressSpace:
    def __init__(self,specs:Sequence[TensorSpec])->None:
        if not specs:
            raise ValueError("AddressSpace requires at least one floating-point tensor")
        self.specs=list(specs)
        self.total_parameters=sum(spec.length for spec in self.specs)
        expected_offset=0
        for spec in self.specs:
            if spec.offset!=expected_offset:
                raise ValueError("TensorSpec offsets must be contiguous")
            expected_offset+=spec.length
        self._offsets=torch.tensor([spec.offset for spec in self.specs],dtype=torch.long)
        self._ends=torch.tensor([spec.offset+spec.length for spec in self.specs],dtype=torch.long)
    @classmethod
    def from_state_dict(cls,state_dict:Mapping[str,Tensor])->AddressSpace:
        specs:list[TensorSpec]=[]
        offset=0
        tensor_id=0
        for name,tensor in state_dict.items():
            if not tensor.is_floating_point():
                continue
            shape=tuple(int(value)for value in tensor.shape)
            length=math.prod(shape)
            specs.append(TensorSpec(name=name,shape=shape,role=infer_parameter_role(name),layer=infer_layer(name),tensor_id=tensor_id,offset=offset,length=length,))
            offset+=length
            tensor_id+=1
        return cls(specs)
    @classmethod
    def from_json(cls,path:str|Path)->AddressSpace:
        raw=json.loads(Path(path).read_text())
        specs=[TensorSpec(name=item["name"],shape=tuple(item["shape"]),role=ParameterRole(item["role"]),layer=item["layer"],tensor_id=item["tensor_id"],offset=item["offset"],length=item["length"],)for item in raw["specs"]]
        return cls(specs)
    def to_json(self,path:str|Path)->None:
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"total_parameters":self.total_parameters,"specs":[{**dataclasses.asdict(spec),"role":int(spec.role),"shape":list(spec.shape),}for spec in self.specs],},indent=2,sort_keys=True,)+"\n")
    def vectorize(self,state_dict:Mapping[str,Tensor])->Tensor:
        values:list[Tensor]=[]
        for spec in self.specs:
            if spec.name not in state_dict:
                raise KeyError(f"Missing tensor {spec.name!r}")
            tensor=state_dict[spec.name]
            if tuple(tensor.shape)!=spec.shape:
                raise ValueError(f"Shape mismatch for {spec.name}: expected {spec.shape}, got {tuple(tensor.shape)}")
            values.append(tensor.reshape(-1))
        return torch.cat(values)
    def unvectorize(self,vector:Tensor,*,template:Mapping[str,Tensor]|None=None,)->OrderedDict[str,Tensor]:
        if vector.numel()!=self.total_parameters:
            raise ValueError(f"Expected vector of length {self.total_parameters}, got {vector.numel()}")
        state:OrderedDict[str,Tensor]=OrderedDict()
        for spec in self.specs:
            value=vector[spec.offset:spec.offset+spec.length].reshape(spec.shape)
            if template is not None and spec.name in template:
                value=value.to(dtype=template[spec.name].dtype,device=template[spec.name].device)
            state[spec.name]=value
        return state
    def sample_indices(self,count:int,*,generator:torch.Generator|None=None,stratified_by_role:bool=True,)->Tensor:
        if count<=0:
            raise ValueError("count must be positive")
        if not stratified_by_role or count<len(self.specs):
            return torch.randint(self.total_parameters,(count,),generator=generator)
        role_to_indices:dict[ParameterRole,list[tuple[int,int]]]={}
        for spec in self.specs:
            role_to_indices.setdefault(spec.role,[]).append((spec.offset,spec.offset+spec.length))
        roles=list(role_to_indices)
        base=count//len(roles)
        remainder=count%len(roles)
        sampled:list[Tensor]=[]
        for position,role in enumerate(roles):
            role_count=base+int(position<remainder)
            spans=role_to_indices[role]
            lengths=torch.tensor([end-start for start,end in spans],dtype=torch.float)
            choices=torch.multinomial(lengths,role_count,replacement=True,generator=generator)
            for choice in choices.tolist():
                start,end=spans[choice]
                sampled.append(torch.randint(start,end,(1,),generator=generator))
        return torch.cat(sampled)[torch.randperm(count,generator=generator)]
    def descriptors(self,indices:Tensor)->dict[str,Tensor]:
        flat_indices=indices.long().reshape(-1).cpu()
        tensor_positions=torch.bucketize(flat_indices,self._ends,right=True)
        if torch.any(tensor_positions>=len(self.specs)):
            raise IndexError("Address index outside parameter vector")
        tensor_ids:list[int]=[]
        roles:list[int]=[]
        layers:list[int]=[]
        continuous:list[list[float]]=[]
        for address,tensor_position in zip(flat_indices.tolist(),tensor_positions.tolist(),strict=True):
            spec=self.specs[tensor_position]
            local_index=address-spec.offset
            coordinates=unravel_index(local_index,spec.shape)
            padded_shape=list(spec.shape[:2])+[1,1]
            padded_coords=list(coordinates[:2])+[0,0]
            row_denominator=max(padded_shape[0]-1,1)
            col_denominator=max(padded_shape[1]-1,1)
            tensor_ids.append(spec.tensor_id)
            roles.append(int(spec.role))
            layers.append(spec.layer+1)
            continuous.append([local_index/max(spec.length-1,1),padded_coords[0]/row_denominator,padded_coords[1]/col_denominator,math.log1p(spec.length)/20.0,math.log1p(padded_shape[0])/12.0,math.log1p(padded_shape[1])/12.0,min(spec.rank,4)/4.0,address/max(self.total_parameters-1,1),])
        original_shape=tuple(indices.shape)
        return{"tensor_id":torch.tensor(tensor_ids,dtype=torch.long).reshape(original_shape),"role_id":torch.tensor(roles,dtype=torch.long).reshape(original_shape),"layer_id":torch.tensor(layers,dtype=torch.long).reshape(original_shape),"continuous":torch.tensor(continuous,dtype=torch.float32).reshape(*original_shape,8),}
    def role_ids_for_all(self)->Tensor:
        role_ids=torch.empty(self.total_parameters,dtype=torch.long)
        for spec in self.specs:
            role_ids[spec.offset:spec.offset+spec.length]=int(spec.role)
        return role_ids
    def indices_for_tensor(self,name:str)->Tensor:
        for spec in self.specs:
            if spec.name==name:
                return torch.arange(spec.offset,spec.offset+spec.length)
        raise KeyError(name)
@dataclasses.dataclass(slots=True)
class WeightStandardizer:
    means:dict[int,float]
    scales:dict[int,float]
    epsilon:float=1e-6
    @classmethod
    def fit(cls,vectors:Tensor,role_ids:Tensor,*,epsilon:float=1e-6,)->WeightStandardizer:
        if vectors.ndim!=2:
            raise ValueError("vectors must have shape [models, parameters]")
        if role_ids.shape!=(vectors.shape[1],):
            raise ValueError("role_ids must have one entry per parameter")
        means:dict[int,float]={}
        scales:dict[int,float]={}
        for role in torch.unique(role_ids).tolist():
            values=vectors[:,role_ids==role].float()
            means[int(role)]=float(values.mean().item())
            scales[int(role)]=max(float(values.std(unbiased=False).item()),epsilon)
        return cls(means=means,scales=scales,epsilon=epsilon)
    @classmethod
    def identity(cls,role_ids:Tensor)->WeightStandardizer:
        roles=[int(role)for role in torch.unique(role_ids).tolist()]
        return cls({role:0.0 for role in roles},{role:1.0 for role in roles})
    def transform(self,values:Tensor,role_ids:Tensor)->Tensor:
        means,scales=self._lookup(role_ids,values.device,values.dtype)
        return(values-means)/scales
    def inverse_transform(self,values:Tensor,role_ids:Tensor)->Tensor:
        means,scales=self._lookup(role_ids,values.device,values.dtype)
        return values*scales+means
    def _lookup(self,role_ids:Tensor,device:torch.device,dtype:torch.dtype)->tuple[Tensor,Tensor]:
        max_role=max(max(self.means,default=0),int(role_ids.max().item()))
        mean_table=torch.zeros(max_role+1,device=device,dtype=dtype)
        scale_table=torch.ones(max_role+1,device=device,dtype=dtype)
        for role,mean in self.means.items():
            mean_table[role]=mean
        for role,scale in self.scales.items():
            scale_table[role]=max(scale,self.epsilon)
        return mean_table[role_ids],scale_table[role_ids]
    def save(self,path:str|Path)->None:
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"means":self.means,"scales":self.scales,"epsilon":self.epsilon},indent=2,sort_keys=True,)+"\n")
    @classmethod
    def load(cls,path:str|Path)->WeightStandardizer:
        raw=json.loads(Path(path).read_text())
        return cls(means={int(key):float(value)for key,value in raw["means"].items()},scales={int(key):float(value)for key,value in raw["scales"].items()},epsilon=float(raw.get("epsilon",1e-6)),)
def infer_parameter_role(name:str)->ParameterRole:
    if "ln" in name or "norm" in name:
        return ParameterRole.NORM_BIAS if name.endswith("bias")else ParameterRole.NORM_WEIGHT
    for suffix,role in ROLE_PATTERNS:
        if name.endswith(suffix):
            return role
    return ParameterRole.OTHER
def infer_layer(name:str)->int:
    match=re.search(r"(?:blocks|layers|h)\.(\d+)\.",name)
    return int(match.group(1))if match else-1
def unravel_index(index:int,shape:tuple[int,...])->tuple[int,...]:
    if not shape:
        return()
    coordinates=[0]*len(shape)
    remainder=index
    for position in range(len(shape)-1,-1,-1):
        coordinates[position]=remainder%shape[position]
        remainder//=shape[position]
    return tuple(coordinates)
