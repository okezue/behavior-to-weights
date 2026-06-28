from __future__ import annotations
import torch
from behavior2weights.models.micro_transformer import MicroTransformer,MicroTransformerConfig
from behavior2weights.zoo.interventions import(ablate_attention_head,ablate_mlp_neuron,apply_lora_edit,)
def test_controlled_interventions_report_changed_coordinates()->None:
    config=MicroTransformerConfig(vocab_size=16,max_seq_len=8,d_model=8,n_heads=2,n_layers=1,d_ff=12)
    state=MicroTransformer(config).ordered_state_dict()
    head=ablate_attention_head(state,config,layer=0,head=1)
    neuron=ablate_mlp_neuron(state,config,layer=0,neuron=3)
    lora=apply_lora_edit(state,tensor_name="blocks.0.attn.o_proj.weight",rank=2,scale=0.1,seed=5)
    assert head.changed_entries>0
    assert neuron.changed_entries>0
    assert lora.changed_entries==state["blocks.0.attn.o_proj.weight"].numel()
    assert not torch.equal(head.state_dict["blocks.0.attn.o_proj.weight"],state["blocks.0.attn.o_proj.weight"])
