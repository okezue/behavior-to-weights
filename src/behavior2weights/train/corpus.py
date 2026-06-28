from __future__ import annotations
import dataclasses
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
import torch
from torch import Tensor
from behavior2weights.models.micro_transformer import(MicroTransformerConfig,canonicalize_state_dict,)
from behavior2weights.models.weight_space import AddressSpace,WeightStandardizer
from behavior2weights.schemas import ObservationChannel,Split,TargetRecord
from behavior2weights.targets.micro import MicroTransformerAdapter
from behavior2weights.traces.store import TraceBundle
@dataclasses.dataclass(slots=True)
class TargetWeights:
    record:TargetRecord
    state_dict:OrderedDict[str,Tensor]
    vector:Tensor
class InverseTrainingCorpus:
    def __init__(self,records:Sequence[TargetRecord],traces:TraceBundle,*,architecture_id:str|None=None,manifest_root:str|Path|None=None,canonicalize:bool=True,)->None:
        trace_index={target_id:index for index,target_id in enumerate(traces.target_ids)}
        selected=[record for record in records if record.target_id in trace_index and(architecture_id is None or record.architecture_id==architecture_id)]
        if not selected:
            raise ValueError("no manifest targets matched the trace bundle and architecture filter")
        architecture_ids={record.architecture_id for record in selected}
        if len(architecture_ids)!=1:
            raise ValueError("coordinate-level recovery requires one architecture_id per corpus; " "train a pooled property classifier separately")
        self.records=selected
        self.traces=traces
        self.trace_indices=torch.tensor([trace_index[record.target_id]for record in selected])
        self.adapter=MicroTransformerAdapter(manifest_root=manifest_root)
        self.targets:list[TargetWeights]=[]
        config:MicroTransformerConfig|None=None
        address_space:AddressSpace|None=None
        for record in selected:
            raw_config=record.metadata.get("model_config")
            if not isinstance(raw_config,dict):
                raise ValueError(f"missing model_config for {record.target_id}")
            current_config=MicroTransformerConfig.from_dict(raw_config)
            if config is None:
                config=current_config
            elif current_config!=config:
                raise ValueError("all targets within an architecture_id must have identical config")
            model=self.adapter.load(record)
            state=OrderedDict((name,tensor.detach().cpu())for name,tensor in model.state_dict().items()if tensor.is_floating_point())
            if canonicalize:
                state=canonicalize_state_dict(state,current_config)
            current_space=AddressSpace.from_state_dict(state)
            if address_space is None:
                address_space=current_space
            elif[spec.name for spec in current_space.specs]!=[spec.name for spec in address_space.specs]or[spec.shape for spec in current_space.specs]!=[spec.shape for spec in address_space.specs]:
                raise ValueError("state dictionaries are incompatible within architecture_id")
            self.targets.append(TargetWeights(record=record,state_dict=state,vector=current_space.vectorize(state)))
        assert config is not None and address_space is not None
        self.model_config=config
        self.address_space=address_space
        self.role_ids=address_space.role_ids_for_all()
        self._record_index={target.record.target_id:index for index,target in enumerate(self.targets)}
    def indices_for_split(self,split:Split)->list[int]:
        return[index for index,target in enumerate(self.targets)if target.record.split==split]
    def fit_standardizer(self,indices:Sequence[int]|None=None)->WeightStandardizer:
        indices=list(indices if indices is not None else self.indices_for_split(Split.TRAIN))
        if not indices:
            raise ValueError("cannot fit weight standardizer without training targets")
        vectors=torch.stack([self.targets[index].vector for index in indices])
        return WeightStandardizer.fit(vectors,self.role_ids)
    def sample_batch(self,target_indices:Sequence[int],*,batch_size:int,query_budget:int,coordinate_count:int,standardizer:WeightStandardizer,generator:torch.Generator,fixed_query_indices:Tensor|None=None,)->dict[str,Tensor|dict[str,Tensor]]:
        if not target_indices:
            raise ValueError("target_indices cannot be empty")
        choices=torch.randint(len(target_indices),(batch_size,),generator=generator)
        selected_indices=[target_indices[index]for index in choices.tolist()]
        if fixed_query_indices is None:
            if query_budget>len(self.traces.query_ids):
                raise ValueError("query_budget exceeds available trace queries")
            query_indices=torch.stack([torch.randperm(len(self.traces.query_ids),generator=generator)[:query_budget]for _ in range(batch_size)])
        else:
            if fixed_query_indices.numel()<query_budget:
                raise ValueError("fixed_query_indices is shorter than query_budget")
            query_indices=fixed_query_indices[:query_budget].repeat(batch_size,1)
        trace_rows=self.trace_indices[torch.tensor(selected_indices)]
        observations=torch.stack([self.traces.observations[row,queries]for row,queries in zip(trace_rows.tolist(),query_indices,strict=True)])
        input_ids=self.traces.input_ids[query_indices]
        channel_id=list(ObservationChannel).index(self.traces.channel)
        channel_ids=torch.full((batch_size,query_budget),channel_id,dtype=torch.long)
        coordinate_indices=self.address_space.sample_indices(coordinate_count,generator=generator,stratified_by_role=True,)
        descriptors=self.address_space.descriptors(coordinate_indices)
        role_ids=descriptors["role_id"]
        targets=torch.stack([self.targets[index].vector[coordinate_indices]for index in selected_indices])
        standardized_targets=standardizer.transform(targets,role_ids)
        return{"input_ids":input_ids,"observations":observations,"channel_ids":channel_ids,"query_mask":torch.ones(batch_size,query_budget,dtype=torch.bool),"descriptors":descriptors,"target_weights":standardized_targets,"coordinate_indices":coordinate_indices,"target_indices":torch.tensor(selected_indices),}
    def target_vector(self,index:int)->Tensor:
        return self.targets[index].vector
