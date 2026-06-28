from __future__ import annotations
import torch
from behavior2weights.models.microtransformer import MicroTransformer,MicroTransformerConfig
from behavior2weights.models.weightspace import AddressSpace,WeightStandardizer
def testaddressspaceroundtripanddescriptors(tmp_path)->None:
    model=MicroTransformer(MicroTransformerConfig(vocab_size=16,max_seq_len=8,d_model=8,n_heads=2,n_layers=1,d_ff=12))
    state=model.orderedstatedict()
    space=AddressSpace.fromstatedict(state)
    vector=space.vectorize(state)
    recovered=space.unvectorize(vector,template=state)
    assert list(recovered)==list(state)
    assert all(torch.equal(state[name],recovered[name])for name in state)
    descriptors=space.descriptors(torch.tensor([0,space.total_parameters-1]))
    assert descriptors["continuous"].shape==(2,8)
    path=tmp_path/"address.json"
    space.tojson(path)
    restored=AddressSpace.fromjson(path)
    assert restored.total_parameters==space.total_parameters
def teststandardizerroundtrip()->None:
    role_ids=torch.tensor([0,0,1,1])
    values=torch.tensor([[1.0,3.0,10.0,14.0],[2.0,4.0,12.0,16.0]])
    standardizer=WeightStandardizer.fit(values,role_ids)
    transformed=standardizer.transform(values,role_ids)
    assert torch.allclose(standardizer.inversetransform(transformed,role_ids),values)
