from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from behavior2weights.stats.inference import(holmadjust,pairedclusterbootstrap,pairedclusterpermutationtest,)
def readresults(paths:list[Path])->pd.DataFrame:
    rows:list[dict[str,object]]=[]
    for path in paths:
        with path.open()as handle:
            rows.extend(json.loads(line)for line in handle if line.strip())
    frame=pd.DataFrame(rows)
    required={"lineage_id","method","metric","value","query_budget","channel","split",}
    missing=required-set(frame)
    if missing:
        raise ValueError(f"result table is missing columns: {sorted(missing)}")
    if "query_policy" not in frame:
        frame["query_policy"]="random"
    return frame
def lineagebudgetauc(frame:pd.DataFrame,*,metric:str,contrast_column:str,levels:tuple[str,str],higher_is_better:bool,)->tuple[pd.DataFrame,list[int],int]:
    subset=frame[(frame["metric"]==metric)&frame[contrast_column].isin(levels)].copy()
    if subset.empty:
        raise ValueError(f"no rows found for metric={metric!r} and levels={levels}")
    subset["query_budget"]=subset["query_budget"].astype(int)
    budgets=sorted(int(value)for value in subset["query_budget"].unique())
    if not budgets:
        raise ValueError("no query budgets are present")
    grouped=(subset.groupby(["lineage_id",contrast_column,"query_budget"],as_index=False)["value"].mean().sort_values(["lineage_id",contrast_column,"query_budget"]))
    expected=set(budgets)
    rows:list[dict[str,object]]=[]
    incomplete=0
    for(lineage_id,level),group in grouped.groupby(["lineage_id",contrast_column],sort=False):
        present=set(int(value)for value in group["query_budget"])
        if present!=expected:
            incomplete+=1
            continue
        ordered=group.set_index("query_budget").loc[budgets]
        values=ordered["value"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            incomplete+=1
            continue
        if not higher_is_better:
            values=-values
        if len(budgets)==1:
            auc=float(values[0])
        else:
            x=np.log2(np.asarray(budgets,dtype=float))
            auc=float(np.trapezoid(values,x=x)/(x[-1]-x[0]))
        rows.append({"lineage_id":str(lineage_id),contrast_column:str(level),"metric":"log_budget_auc","value":auc,})
    return pd.DataFrame(rows),budgets,incomplete
def paireddifferences(summary:pd.DataFrame,*,contrast_column:str,level_a:str,level_b:str,)->pd.Series:
    paired=summary.pivot(index="lineage_id",columns=contrast_column,values="value")
    if level_a not in paired or level_b not in paired:
        raise ValueError("both contrast levels must be present after lineage aggregation")
    paired=paired[[level_a,level_b]].dropna()
    return paired[level_a]-paired[level_b]
def main()->None:
    parser=argparse.ArgumentParser(description="Frozen lineage-level query-efficiency aggregation")
    parser.add_argument("results",nargs="+",type=Path)
    parser.add_argument("--contrast",choices=("method","query_policy"),default="method",help="Field defining the paired contrast.",)
    parser.add_argument("--level-a")
    parser.add_argument("--level-b")
    parser.add_argument("--fixed-method",default="inverse_posterior_mean",help="Method retained when contrasting query policies.",)
    parser.add_argument("--fixed-query-policy",default="random",help="Query policy retained when contrasting methods.",)
    parser.add_argument("--metric",action="append",required=True,help="metric[:higher|lower]")
    parser.add_argument("--resamples",type=int,default=10_000)
    parser.add_argument("--permutations",type=int,default=100_000)
    parser.add_argument("--alternative",choices=("two-sided","greater","less"),default="two-sided")
    parser.add_argument("--seed",type=int,default=20260621)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    frame=readresults(args.results)
    if args.contrast=="method":
        level_a=args.level_a or "inverse_posterior_mean"
        level_b=args.level_b or "mean_checkpoint"
        frame=frame[frame["query_policy"]==args.fixed_query_policy]
        fixed={"query_policy":args.fixed_query_policy}
    else:
        level_a=args.level_a or "population_disagreement"
        level_b=args.level_b or "random"
        frame=frame[frame["method"]==args.fixed_method]
        fixed={"method":args.fixed_method}
    analyses:list[dict[str,object]]=[]
    for index,specification in enumerate(args.metric):
        parts=specification.rsplit(":",1)
        metric=parts[0]
        direction=parts[1]if len(parts)==2 else "higher"
        if direction not in{"higher","lower"}:
            raise ValueError("metric direction must be higher or lower")
        higher=direction=="higher"
        summary,budgets,incomplete=lineagebudgetauc(frame,metric=metric,contrast_column=args.contrast,levels=(level_a,level_b),higher_is_better=higher,)
        differences=paireddifferences(summary,contrast_column=args.contrast,level_a=level_a,level_b=level_b,)
        estimate=pairedclusterbootstrap(summary,method_a=level_a,method_b=level_b,metric="log_budget_auc",method_column=args.contrast,resamples=args.resamples,seed=args.seed+index,)
        p_value=pairedclusterpermutationtest(differences,permutations=args.permutations,alternative=args.alternative,seed=args.seed+1_000+index,)
        analyses.append({"metric":metric,"direction":direction,"estimand":"mean paired difference in log2-query-budget AUC","contrast_field":args.contrast,"level_a":level_a,"level_b":level_b,"fixed":fixed,"budget_grid":budgets,"estimate_a_advantage":estimate.estimate,"ci_lower":estimate.lower,"ci_upper":estimate.upper,"standard_error":estimate.standard_error,"paired_lineages":estimate.n_clusters,"incomplete_lineage_conditions":incomplete,"p_value":p_value,"alternative":args.alternative,})
    adjusted=holmadjust([float(row["p_value"])for row in analyses])
    for row,value in zip(analyses,adjusted,strict=True):
        row["p_value_holm"]=value
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"analyses":analyses},indent=2,sort_keys=True)+"\n")
    pd.DataFrame(analyses).to_csv(args.output.with_suffix(".csv"),index=False)
    print(args.output)
if __name__=="__main__":
    main()
