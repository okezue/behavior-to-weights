from __future__ import annotations
import torch
from behavior2weights.schemas import ObservationChannel
from behavior2weights.traces.store import TraceBundle,loadtracebundle,savetracebundle
def testtracestoreroundtripandchecksum(tmp_path)->None:
    bundle=TraceBundle(target_ids=("a","b"),query_ids=("q1","q2","q3"),input_ids=torch.arange(12).reshape(3,4),observations=torch.randn(2,3,5),channel=ObservationChannel.LOGITS,feature_dim=5,metadata={"test":True},)
    savetracebundle(bundle,tmp_path)
    loaded=loadtracebundle(tmp_path)
    assert loaded.target_ids==bundle.target_ids
    assert loaded.query_ids==bundle.query_ids
    assert torch.equal(loaded.input_ids,bundle.input_ids)
    assert torch.equal(loaded.observations,bundle.observations)
def testtracestoreroundtripsauxiliaryoutputs(tmp_path)->None:
    from behavior2weights.traces.store import TraceBundle,loadtracebundle,savetracebundle
    bundle=TraceBundle(target_ids=("t0","t1"),query_ids=("q0","q1","q2"),input_ids=torch.arange(12).reshape(3,4),observations=torch.randn(2,3,5),channel=ObservationChannel.TOPK,feature_dim=5,auxiliary={"topk_indices":torch.randint(0,5,(2,3,2)),"topk_values":torch.randn(2,3,2),},)
    savetracebundle(bundle,tmp_path)
    loaded=loadtracebundle(tmp_path)
    assert set(loaded.auxiliary)=={"topk_indices","topk_values"}
    assert torch.equal(loaded.auxiliary["topk_indices"],bundle.auxiliary["topk_indices"])
    assert torch.equal(loaded.subsettargets([1]).auxiliary["topk_values"],bundle.auxiliary["topk_values"][1:2],)
