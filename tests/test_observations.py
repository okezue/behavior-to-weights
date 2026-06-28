from __future__ import annotations
import pytest
import torch
from behavior2weights.schemas import ObservationChannel
from behavior2weights.traces.observations import ObservationConfig,encode_logits
@pytest.mark.parametrize("channel",list(ObservationChannel))
def test_observation_channels_have_fixed_width(channel:ObservationChannel)->None:
    logits=torch.randn(5,16)
    config=ObservationConfig(channel=channel,vocab_size=16,feature_dim=16,topk=4,sample_count=20,sketch_dim=8,)
    first,metadata=encode_logits(logits,config,generator=torch.Generator().manual_seed(3))
    assert first.shape==(5,16)
    assert torch.isfinite(first).all()
    if channel==ObservationChannel.SAMPLE_HISTOGRAM:
        assert torch.allclose(first.sum(dim=-1),torch.ones(5))
    if channel==ObservationChannel.TOKENS:
        assert torch.equal(first.sum(dim=-1),torch.ones(5))
    if channel==ObservationChannel.TOPK:
        assert metadata["topk_indices"].shape==(5,4)
def test_sampling_channel_is_seed_reproducible()->None:
    logits=torch.randn(4,12)
    config=ObservationConfig(channel=ObservationChannel.TOKENS,vocab_size=12,feature_dim=12)
    a,_=encode_logits(logits,config,generator=torch.Generator().manual_seed(9))
    b,_=encode_logits(logits,config,generator=torch.Generator().manual_seed(9))
    assert torch.equal(a,b)
def test_trace_sampling_is_invariant_to_target_and_query_order()->None:
    from pathlib import Path
    from torch import nn
    from behavior2weights.schemas import QueryRecord,TargetRecord
    from behavior2weights.traces.collector import CollectionConfig,collect_target_traces
    class FixedModel(nn.Module):
        def forward(self,input_ids:torch.Tensor)->torch.Tensor:
            batch,sequence=input_ids.shape
            logits=torch.linspace(-1.0,1.0,8).repeat(batch,sequence,1)
            return logits
    targets=[TargetRecord(target_id=target_id,family_id="test",lineage_id=target_id,architecture_id="fixed",task_id="test",dataset_id="test",seed=0,checkpoint_path=Path("unused"),)for target_id in("target-a","target-b")]
    queries=[QueryRecord(query_id=f"query-{index}",input_ids=[index,1,2],source="test",)for index in range(3)]
    observation=ObservationConfig(channel=ObservationChannel.SAMPLE_HISTOGRAM,vocab_size=8,feature_dim=8,sample_count=17,)
    collection=CollectionConfig(base_seed=19)
    first=collect_target_traces(targets,queries,lambda _:FixedModel(),observation,collection,)
    second=collect_target_traces(list(reversed(targets)),list(reversed(queries)),lambda _:FixedModel(),observation,collection,)
    for target_id in first.target_ids:
        first_target=first.target_ids.index(target_id)
        second_target=second.target_ids.index(target_id)
        for query_id in first.query_ids:
            first_query=first.query_ids.index(query_id)
            second_query=second.query_ids.index(query_id)
            assert torch.equal(first.observations[first_target,first_query],second.observations[second_target,second_query],)
def test_observation_config_from_experiment_channel_spec()->None:
    config=ObservationConfig.from_dict({"name":"logit_sketch","sketch_dim":7,"sketch_seed":99},vocab_size=32,)
    assert config.channel==ObservationChannel.LOGIT_SKETCH
    assert config.feature_dim==7
    assert config.sketch_seed==99
    with pytest.raises(ValueError,match="Unknown ObservationConfig fields"):
        ObservationConfig.from_dict({"name":"logits","typo":1},vocab_size=32)
def test_compact_channels_preserve_high_token_ids_in_auxiliary_metadata()->None:
    logits=torch.full((2,1000),-20.0)
    logits[:,999]=20.0
    token_config=ObservationConfig(channel=ObservationChannel.TOKENS,vocab_size=1000,feature_dim=32,)
    token_observation,token_metadata=encode_logits(logits,token_config,generator=torch.Generator().manual_seed(1))
    assert torch.equal(token_metadata["output_ids"],torch.full((2,1),999))
    assert torch.equal(token_observation.argmax(dim=-1),torch.full((2,),999%32))
    topk_config=ObservationConfig.from_dict({"name":"topk","topk":4},vocab_size=1000)
    topk_observation,topk_metadata=encode_logits(logits,topk_config)
    assert topk_observation.shape==(2,8)
    assert topk_metadata["topk_indices"][0,0].item()==999
def test_full_logits_cannot_be_silently_truncated()->None:
    with pytest.raises(ValueError,match="full-logit observations"):
        ObservationConfig(channel=ObservationChannel.LOGITS,vocab_size=100,feature_dim=32,)
