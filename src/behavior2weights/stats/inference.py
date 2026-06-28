from __future__ import annotations
import dataclasses
from collections.abc import Sequence
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
@dataclasses.dataclass(frozen=True,slots=True)
class Estimate:
    estimate:float
    lower:float
    upper:float
    p_value:float|None=None
    standard_error:float|None=None
    n_clusters:int|None=None
    method:str=""
def _paired_cluster_values(frame:pd.DataFrame,*,method_a:str,method_b:str,metric:str,cluster_column:str,method_column:str,metric_column:str,value_column:str,higher_is_better:bool,)->pd.Series:
    subset=frame[frame[metric_column]==metric]
    grouped=(subset.groupby([cluster_column,method_column],as_index=False)[value_column].mean().pivot(index=cluster_column,columns=method_column,values=value_column))
    if method_a not in grouped or method_b not in grouped:
        raise ValueError("both methods must be present")
    paired=grouped[[method_a,method_b]].dropna()
    difference=paired[method_a]-paired[method_b]
    return difference if higher_is_better else-difference
def paired_cluster_bootstrap(frame:pd.DataFrame,*,method_a:str,method_b:str,metric:str,cluster_column:str="lineage_id",method_column:str="method",metric_column:str="metric",value_column:str="value",higher_is_better:bool=True,resamples:int=10_000,confidence:float=0.95,seed:int=0,)->Estimate:
    differences=_paired_cluster_values(frame,method_a=method_a,method_b=method_b,metric=metric,cluster_column=cluster_column,method_column=method_column,metric_column=metric_column,value_column=value_column,higher_is_better=higher_is_better,).to_numpy()
    if len(differences)<2:
        raise ValueError("at least two paired clusters are required")
    generator=np.random.default_rng(seed)
    draws=generator.choice(differences,size=(resamples,len(differences)),replace=True).mean(axis=1)
    alpha=1-confidence
    lower,upper=np.quantile(draws,[alpha/2,1-alpha/2])
    return Estimate(estimate=float(differences.mean()),lower=float(lower),upper=float(upper),standard_error=float(draws.std(ddof=1)),n_clusters=len(differences),method="paired lineage-cluster bootstrap",)
def paired_cluster_permutation_test(differences:Sequence[float],*,permutations:int=100_000,alternative:str="two-sided",seed:int=0,)->float:
    values=np.asarray(differences,dtype=float)
    if values.ndim!=1 or len(values)<2:
        raise ValueError("differences must contain at least two paired cluster estimates")
    observed=values.mean()
    generator=np.random.default_rng(seed)
    signs=generator.choice([-1.0,1.0],size=(permutations,len(values)))
    null=(signs*values).mean(axis=1)
    if alternative=="greater":
        extreme=null>=observed
    elif alternative=="less":
        extreme=null<=observed
    elif alternative=="two-sided":
        extreme=np.abs(null)>=abs(observed)
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return float((extreme.sum()+1)/(permutations+1))
def holm_adjust(p_values:Sequence[float])->list[float]:
    values=np.asarray(p_values,dtype=float)
    order=np.argsort(values)
    adjusted=np.empty_like(values)
    running=0.0
    count=len(values)
    for rank,index in enumerate(order):
        running=max(running,(count-rank)*values[index])
        adjusted[index]=min(running,1.0)
    return[float(value)for value in adjusted]
def standardized_paired_effect(differences:Sequence[float])->float:
    values=np.asarray(differences,dtype=float)
    return float(values.mean()/values.std(ddof=1))
def one_sample_summary(values:Sequence[float],confidence:float=0.95)->Estimate:
    array=np.asarray(values,dtype=float)
    mean=array.mean()
    sem=stats.sem(array)
    quantile=stats.t.ppf((1+confidence)/2,df=len(array)-1)
    return Estimate(estimate=float(mean),lower=float(mean-quantile*sem),upper=float(mean+quantile*sem),standard_error=float(sem),n_clusters=len(array),method="Student-t interval",)
def fit_mixed_effects(frame:pd.DataFrame,*,formula:str,group_column:str="lineage_id",re_formula:str="1",)->Any:
    try:
        import statsmodels.formula.api as smf
    except ImportError as error:
        raise RuntimeError("statsmodels is required for mixed-effects analysis")from error
    model=smf.mixedlm(formula,frame,groups=frame[group_column],re_formula=re_formula)
    return model.fit(method="lbfgs",reml=False)
