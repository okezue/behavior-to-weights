from __future__ import annotations
import torch
from behavior2weights.models.micro_transformer import MicroTransformer,MicroTransformerConfig
from behavior2weights.models.weight_space import AddressSpace,WeightStandardizer
def test_address_space_roundtrip_and_descriptors(tmp_path)->None:
    model=MicroTransformer(MicroTransformerConfig(vocab_size=16,max_seq_len=8,d_model=8,n_heads=2,n_layers=1,d_ff=12))
    state=model.ordered_state_dict()
    space=AddressSpace.from_state_dict(state)
    vector=space.vectorize(state)
    recovered=space.unvectorize(vector,template=state)
    assert list(recovered)==list(state)
    assert all(torch.equal(state[name],recovered[name])for name in state)
    descriptors=space.descriptors(torch.tensor([0,space.total_parameters-1]))
    assert descriptors["continuous"].shape==(2,8)
    path=tmp_path/"address.json"
    space.to_json(path)
    restored=AddressSpace.from_json(path)
    assert restored.total_parameters==space.total_parameters
def test_standardizer_roundtrip()->None:
    role_ids=torch.tensor([0,0,1,1])
    values=torch.tensor([[1.0,3.0,10.0,14.0],[2.0,4.0,12.0,16.0]])
    standardizer=WeightStandardizer.fit(values,role_ids)
    transformed=standardizer.transform(values,role_ids)
    assert torch.allclose(standardizer.inverse_transform(transformed,role_ids),values)
