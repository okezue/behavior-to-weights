# References and Related Work

This is a working bibliography; pin exact versions used in a paper.

## Parameter and model extraction

- Carlini et al., **Stealing Part of a Production Language Model**, 2024. https://arxiv.org/abs/2403.06634
- Jagielski et al., work on high-fidelity neural-network extraction and parameter recovery.
- Expand-and-cluster style exact parameter-recovery methods for shallow networks.
- Recent input/output-query reverse-engineering and teacher–student recovery studies; verify final publication metadata before citation.

## Weight-space and hypernetwork methods

- Ha, Dai, and Le, **HyperNetworks**, 2016. https://arxiv.org/abs/1609.09106
- Neural Functional Transformers, NeurIPS 2023.
- Martinelli et al., **Transformer Neural Functional Networks**, ICML 2024. https://proceedings.mlr.press/v235/martinelli24a.html
- Dataset-to-parameter hypernetwork and parameter-generation work, including ICML 2025 studies; use final proceedings metadata.

## Interpretability

- Anthropic, **Circuit Tracing: Revealing Computational Graphs in Language Models**, 2025. https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Transformer Circuits work on induction heads, superposition, and attribution/causal interventions.
- Activation patching, causal tracing, sparse autoencoders, and QK/attention tracing literature.

## Model populations

- Biderman et al., **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling**, 2023. Public model cards expose 154 checkpoints per size.
- Eldan and Li, **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?**, 2023. https://arxiv.org/abs/2305.07759
- Allal et al., **SmolLM2: When Smol Goes Big**, 2025. https://arxiv.org/abs/2502.02737
- Transformer-NFN Small Transformer Zoo: https://github.com/Fsoft-AIC/Transformer-NFN

## Experiment tracking and data tooling

- AIM: https://github.com/aimhubio/aim and https://aimstack.readthedocs.io/
- Hugging Face Datasets and Transformers official documentation.

## Statistical guidance

- Clustered/hierarchical bootstrap and randomization inference references appropriate to the final venue.
- Mixed-effects modeling references; report convergence and small-cluster limitations.
- Multiple-testing control: Holm familywise correction and Benjamini–Hochberg FDR.
