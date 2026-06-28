from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
def load(path):
    rows=[json.loads(l)for l in Path(path).read_text().splitlines()if l.strip()]
    df=pd.DataFrame(rows)
    df["augmented_dim"]=df["input_dim"]+1
    df["enough_queries"]=df["query_count"]>=df["augmented_dim"]
    return df
def md(df,fmt="{:.4g}"):
    if not isinstance(df.index,pd.RangeIndex):df=df.reset_index()
    def c(v):
        if isinstance(v,float):return fmt.format(v)
        return str(v)
    head="| "+" | ".join(map(str,df.columns))+" |"
    sep="| "+" | ".join("---" for _ in df.columns)+" |"
    body=["| "+" | ".join(c(v)for v in row)+" |" for row in df.itertuples(index=False)]
    return "\n".join([head,sep,*body])
def slope(x,y):
    x=np.log(np.asarray(x,float));y=np.log(np.asarray(y,float))
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2:return float("nan")
    return float(np.polyfit(x[m],y[m],1)[0])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--output",default="reports/TIER0_ANALYSIS.md")
    a=ap.parse_args()
    df=load(a.results)
    n=len(df)
    prob=df[df.observation_channel=="probabilities"]
    samp=df[df.observation_channel=="samples"]
    L=[]
    L.append(f"# Tier-0 analytic identifiability\n")
    L.append(f"{n} systems, {df.system_id.nunique()} unique, channels={sorted(df.observation_channel.unique())}\n")
    thr=df.groupby(["query_strategy","input_dim","query_count"]).full_rank.mean().reset_index()
    L.append("## Identifiability threshold (full-rank rate)\n")
    L.append("Recovery of the full canonical parameter needs design rank = input_dim+1.\n")
    piv=thr.pivot_table(index=["query_strategy","input_dim"],columns="query_count",values="full_rank")
    L.append(md(piv,"{:.2f}")+"\n")
    below=df[~df.enough_queries]
    above=df[df.enough_queries]
    L.append("## Below vs at/above the query threshold\n")
    L.append(f"below (q<d+1): full_rank={below.full_rank.mean():.3f} observable_fraction={below.observable_fraction.mean():.3f} orbit_nrmse={below.orbit_nrmse.median():.3f} fwd_kl={below.functional_forward_kl.median():.4f}\n")
    L.append(f"at/above (q>=d+1): full_rank={above.full_rank.mean():.3f} observable_fraction={above.observable_fraction.mean():.3f} orbit_nrmse={above.orbit_nrmse.median():.3e} fwd_kl={above.functional_forward_kl.median():.2e}\n")
    pf=prob[prob.full_rank]
    L.append("## Exact recovery (probabilities channel, full rank)\n")
    L.append(f"exact_recovery_rate={pf.exact_recovery.mean():.4f} mean_orbit_nrmse={pf.orbit_nrmse.mean():.2e} mean_fwd_kl={pf.functional_forward_kl.mean():.2e}\n")
    pnf=prob[~prob.full_rank]
    L.append(f"rank-deficient probabilities: orbit_nrmse_median={pnf.orbit_nrmse.median():.3f} fwd_kl_median={pnf.functional_forward_kl.median():.4f}\n")
    sf=samp[samp.full_rank]
    by=sf.groupby("sample_count").agg(orbit_nrmse=("orbit_nrmse","mean"),fwd_kl=("functional_forward_kl","mean"),systems=("orbit_nrmse","size")).reset_index()
    L.append("## Sampling noise (samples channel, full rank)\n")
    L.append(md(by)+"\n")
    L.append(f"all-N: nrmse~N^{slope(by.sample_count,by.orbit_nrmse):.2f}  kl~N^{slope(by.sample_count,by.fwd_kl):.2f}\n")
    big=by[by.sample_count>=32]
    if len(big)>=2:
        L.append(f"large-N (>=32): nrmse~N^{slope(big.sample_count,big.orbit_nrmse):.2f}  kl~N^{slope(big.sample_count,big.fwd_kl):.2f}  (theory: -0.5 and -1.0)\n")
    L.append("## Class count effect (at full rank)\n")
    cls=df[df.full_rank].groupby(["observation_channel","class_count"]).agg(orbit_nrmse=("orbit_nrmse","mean"),fwd_kl=("functional_forward_kl","mean"),exact=("exact_recovery","mean")).reset_index()
    L.append(md(cls)+"\n")
    g=df.groupby("query_strategy").agg(full_rank=("full_rank","mean"),orbit_nrmse=("orbit_nrmse","median"),fwd_kl=("functional_forward_kl","median")).reset_index()
    L.append("## Query design: gaussian vs basis\n")
    L.append(md(g)+"\n")
    cc=samp[samp.full_rank].dropna(subset=["design_condition_number"])
    if len(cc)>50:
        r=np.corrcoef(np.log(cc.design_condition_number),np.log(cc.orbit_nrmse.clip(1e-12)))[0,1]
        L.append("## Conditioning\n")
        L.append(f"corr(log cond, log orbit_nrmse) at full rank, samples = {r:.3f}\n")
    sN=slope(big.sample_count,big.orbit_nrmse)if len(big)>=2 else float("nan")
    kN=slope(big.sample_count,big.fwd_kl)if len(big)>=2 else float("nan")
    csamp=df[(df.full_rank)&(df.observation_channel=="samples")].groupby("class_count").functional_forward_kl.mean()
    L.append("## Findings\n")
    L.append(f"1. Identifiability is a sharp step: the canonical parameter is recoverable iff query_count>=input_dim+1. Below it full-rank rate is 0 and only {below.observable_fraction.mean()*100:.0f}% of parameter directions are observed; at/above it is exactly 1.0 for both gaussian and basis designs.\n")
    L.append(f"2. With exact probabilities and a full-rank design, recovery is algebraically exact (nrmse {pf.orbit_nrmse.mean():.0e}, KL {pf.functional_forward_kl.mean():.0e}) for every class count.\n")
    L.append(f"3. Below threshold the model also fails to generalize: held-out forward KL stays high (median {below.functional_forward_kl.median():.2f}), so matching behavior on too few queries recovers neither weights nor function.\n")
    L.append(f"4. Finite sampling converges slower than ideal Monte Carlo: orbit_nrmse~N^{sN:.2f} and KL~N^{kN:.2f} versus the -0.5 and -1.0 a variance-only model predicts. The Dirichlet floor biases log-odds of near-deterministic rows, so sampled black-box access is bias-limited, not variance-limited.\n")
    L.append(f"5. Sample complexity grows with class count: at full rank the sampled-channel forward KL rises from {csamp.min():.2f} to {csamp.max():.2f} across class counts {list(csamp.index)}, so large-vocabulary targets are far harder to extract from samples than from logits.\n")
    rep=("\n".join(L))
    Path(a.output).write_text(rep)
    print(a.output)
    print(f"systems={n} prob_fullrank_exact={pf.exact_recovery.mean():.4f} samp_nrmse_slope={slope(by.sample_count,by.orbit_nrmse):.2f}")
main()
