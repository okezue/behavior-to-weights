from __future__ import annotations
import dataclasses
import numpy as np
from scipy import stats
@dataclasses.dataclass(frozen=True,slots=True)
class PowerResult:
    lineages:int
    checkpoints_per_lineage:int
    effect_size:float
    intraclass_correlation:float
    power:float
    simulations:int
def analyticpairedsamplesize(standardized_effect:float,*,alpha:float=0.05,power:float=0.80,two_sided:bool=True,)->int:
    if standardized_effect<=0:
        raise ValueError("standardized_effect must be positive")
    alpha_quantile=stats.norm.ppf(1-alpha/(2 if two_sided else 1))
    power_quantile=stats.norm.ppf(power)
    return int(np.ceil(((alpha_quantile+power_quantile)/standardized_effect)**2))
def simulateclusteredpower(*,lineages:int,checkpoints_per_lineage:int,standardized_effect:float,intraclass_correlation:float=0.5,alpha:float=0.05,simulations:int=5_000,seed:int=0,)->PowerResult:
    if not 0<=intraclass_correlation<1:
        raise ValueError("intraclass_correlation must be in [0, 1)")
    generator=np.random.default_rng(seed)
    rejections=0
    cluster_sd=intraclass_correlation**0.5
    residual_sd=(1-intraclass_correlation)**0.5
    for _ in range(simulations):
        lineage_effect=generator.normal(0,cluster_sd,size=(lineages,1))
        residual=generator.normal(0,residual_sd,size=(lineages,checkpoints_per_lineage))
        observations=standardized_effect+lineage_effect+residual
        cluster_means=observations.mean(axis=1)
        _,p_value=stats.ttest_1samp(cluster_means,0)
        rejections+=p_value<alpha
    return PowerResult(lineages=lineages,checkpoints_per_lineage=checkpoints_per_lineage,effect_size=standardized_effect,intraclass_correlation=intraclass_correlation,power=rejections/simulations,simulations=simulations,)
def findminimumlineages(*,standardized_effect:float,checkpoints_per_lineage:int=1,intraclass_correlation:float=0.5,target_power:float=0.8,alpha:float=0.05,min_lineages:int=8,max_lineages:int=512,simulations:int=2_000,seed:int=0,)->PowerResult:
    last:PowerResult|None=None
    for lineages in range(min_lineages,max_lineages+1,4):
        last=simulateclusteredpower(lineages=lineages,checkpoints_per_lineage=checkpoints_per_lineage,standardized_effect=standardized_effect,intraclass_correlation=intraclass_correlation,alpha=alpha,simulations=simulations,seed=seed+lineages,)
        if last.power>=target_power:
            return last
    assert last is not None
    return last
