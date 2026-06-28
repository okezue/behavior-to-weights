from __future__ import annotations
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from behavior2weights.probes.active import training_population_order
from behavior2weights.schemas import ResultRecord,Split
from behavior2weights.traces.store import load_trace_bundle
from behavior2weights.train.property import TracePropertyCorpus,load_property_checkpoint
from behavior2weights.utils import stable_hash,write_jsonl
from behavior2weights.zoo.manifest import load_manifest
@torch.no_grad()
def evaluate_property_classifier(*,manifest_path:str|Path,traces_directory:str|Path,checkpoint_directory:str|Path,output_path:str|Path,query_budgets:tuple[int,...]=(16,32,64),query_policies:tuple[str,...]=("random","population_disagreement"),splits:tuple[Split,...]=(Split.TEST,Split.OOD),tier:str="tier2",run_id:str="property-evaluation",replicate:int=0,seed:int=20260621,device:str="cpu",)->list[ResultRecord]:
    records=load_manifest(manifest_path,resolve_paths=False)
    traces=load_trace_bundle(traces_directory)
    model,vocabulary=load_property_checkpoint(checkpoint_directory,device=device)
    properties=tuple(vocabulary.values)
    corpus=TracePropertyCorpus(records,traces,properties=properties,vocabulary=vocabulary)
    indices=[index for split in splits for index in corpus.indices_for_split(split)]
    if not indices:
        raise ValueError("no targets in requested evaluation splits")
    budgets=tuple(sorted(set(query_budgets)))
    if not budgets or min(budgets)<=0 or max(budgets)>len(traces.query_ids):
        raise ValueError("query budgets must be positive and fit the trace bank")
    policies=tuple(dict.fromkeys(query_policies))
    if not policies:
        raise ValueError("query_policies cannot be empty")
    train_indices=corpus.indices_for_split(Split.TRAIN)
    training_rows=corpus.trace_indices[torch.tensor(train_indices)]
    training_lineages=[corpus.records[index].lineage_id for index in train_indices]
    policy_orders={policy:training_population_order(traces.observations[training_rows],training_lineages,policy=policy,seed=seed+policy_index*1_000_003,).indices for policy_index,policy in enumerate(policies)}
    model.eval()
    output:list[ResultRecord]=[]
    for query_policy in policies:
        order=policy_orders[query_policy]
        for budget in budgets:
            queries=order[:budget]
            query_hash=stable_hash(queries.tolist(),length=32)
            batch=corpus.batch(indices,query_indices=queries)
            logits=model(batch["input_ids"].to(device),batch["observations"].to(device),batch["channel_ids"].to(device),query_mask=batch["query_mask"].to(device),)
            for name,values in logits.items():
                probabilities=values.softmax(dim=-1).cpu()
                labels=batch["labels"][name]
                predictions=probabilities.argmax(dim=-1)
                losses=F.cross_entropy(values.cpu(),labels,reduction="none")
                entropy_denominator=math.log(probabilities.shape[-1])
                for row,target_index in enumerate(indices):
                    record=corpus.records[target_index]
                    metadata={"property":name,"true_label":vocabulary.decode(name,int(labels[row])),"predicted_label":vocabulary.decode(name,int(predictions[row])),"unknown_target":bool(labels[row]==0),"query_index_hash":query_hash,"query_policy_fit_split":"train",}
                    entropy=-(probabilities[row]*probabilities[row].clamp_min(1e-12).log()).sum()
                    metrics={f"property_{name}_accuracy":float(predictions[row]==labels[row]),f"property_{name}_nll":float(losses[row]),f"property_{name}_confidence":float(probabilities[row].max()),f"property_{name}_entropy":float(entropy/entropy_denominator if entropy_denominator>0 else 0.0),}
                    for metric,value in metrics.items():
                        output.append(ResultRecord(run_id=run_id,target_id=record.target_id,lineage_id=record.lineage_id,split=record.split or Split.TEST,tier=tier,method="trace_property_classifier",channel=traces.channel,query_policy=query_policy,query_budget=budget,replicate=replicate,metric=metric,value=value,metadata=metadata,))
    write_jsonl(output_path,[row.model_dump(mode="json")for row in output])
    return output
