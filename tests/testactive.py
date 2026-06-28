from __future__ import annotations
import torch
from behavior2weights.probes.active import trainingpopulationorder
def testpopulationdisagreementuseslineagesnottargetrows()->None:
    observations=torch.tensor([[[0.0],[0.0],[1.0]],[[0.0],[100.0],[1.0]],[[10.0],[50.0],[1.0]],])
    ranked=trainingpopulationorder(observations,["lineage-a","lineage-a","lineage-b"],policy="population_disagreement",seed=7,)
    assert int(ranked.indices[0])==0
    assert ranked.utilities[0]>ranked.utilities[1]
def testrandomcandidateorderisreproducibleandcomplete()->None:
    observations=torch.zeros(2,7,3)
    first=trainingpopulationorder(observations,["a","b"],policy="random",seed=11)
    second=trainingpopulationorder(observations,["a","b"],policy="random",seed=11)
    assert torch.equal(first.indices,second.indices)
    assert sorted(first.indices.tolist())==list(range(7))
