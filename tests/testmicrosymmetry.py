from __future__ import annotations
import torch
from behavior2weights.models.microtransformer import(MicroTransformer,MicroTransformerConfig,canonicalizestatedict,permuteattentionheads,permutemlpneurons,)
def testheadandneuronpermutationspreservefunction()->None:
    torch.manual_seed(7)
    config=MicroTransformerConfig(vocab_size=16,max_seq_len=8,d_model=8,n_heads=2,n_layers=1,d_ff=12)
    reference=MicroTransformer(config).eval()
    state=reference.orderedstatedict()
    permuted=permuteattentionheads(state,config,0,torch.tensor([1,0]))
    permuted=permutemlpneurons(permuted,0,torch.arange(config.d_ff-1,-1,-1))
    candidate=MicroTransformer(config).eval()
    candidate.load_state_dict(permuted)
    inputs=torch.randint(config.vocab_size,(17,config.max_seq_len-1))
    assert torch.allclose(reference(inputs),candidate(inputs),atol=1e-6,rtol=1e-5)
def testcanonicalizationmatchespermutedorbit()->None:
    torch.manual_seed(11)
    config=MicroTransformerConfig(vocab_size=16,max_seq_len=8,d_model=8,n_heads=2,n_layers=1,d_ff=10)
    state=MicroTransformer(config).orderedstatedict()
    permuted=permuteattentionheads(state,config,0,torch.tensor([1,0]))
    permuted=permutemlpneurons(permuted,0,torch.randperm(config.d_ff))
    first=canonicalizestatedict(state,config)
    second=canonicalizestatedict(permuted,config)
    assert first.keys()==second.keys()
    for name in first:
        assert torch.equal(first[name],second[name]),name
