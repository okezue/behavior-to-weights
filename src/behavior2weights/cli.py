from __future__ import annotations
import dataclasses
import json
from pathlib import Path
from typing import Annotated
import typer
from behavior2weights.analytic.softmax import(LinearSoftmaxExperimentConfig,run_linear_softmax_experiment,)
from behavior2weights.compute.runtime import runtime_info_dict
from behavior2weights.compute.smoke import run_smoke_pipeline
from behavior2weights.config import TrackingConfig
from behavior2weights.data.hf_text import PRESETS,HFTextDatasetConfig,prepare_hf_causal_lm_dataset
from behavior2weights.data.tokenizer import TokenizerTrainingConfig,train_byte_bpe_tokenizer
from behavior2weights.evaluation.property_runner import evaluate_property_classifier
from behavior2weights.evaluation.runner import EvaluationConfig,evaluate_micro_inverse
from behavior2weights.models.inverse import BehaviorToWeights,InverseModelConfig
from behavior2weights.models.property import PropertyModelConfig
from behavior2weights.probes.hf_query_bank import build_hf_query_bank
from behavior2weights.probes.query_bank import QueryBankConfig,build_query_bank,save_query_bank
from behavior2weights.schemas import ObservationChannel,QueryRecord,Split
from behavior2weights.stats.power import analytic_paired_sample_size,simulate_clustered_power
from behavior2weights.targets.huggingface import HuggingFaceCausalLMAdapter
from behavior2weights.targets.micro import MicroTransformerAdapter
from behavior2weights.traces.collector import CollectionConfig,collect_target_traces
from behavior2weights.traces.observations import ObservationConfig
from behavior2weights.traces.store import load_trace_bundle
from behavior2weights.tracking.base import ExperimentTracker
from behavior2weights.tracking.factory import create_tracker
from behavior2weights.train.corpus import InverseTrainingCorpus
from behavior2weights.train.inverse import InverseTrainingConfig,train_inverse_model
from behavior2weights.train.property import(PropertyTrainingConfig,TracePropertyCorpus,train_property_classifier,)
from behavior2weights.utils import load_yaml,read_jsonl
from behavior2weights.zoo.external import download_transformer_nfn_zoo
from behavior2weights.zoo.manifest import SplitPolicy,load_manifest,manifest_summary
from behavior2weights.zoo.micro import MicroZooConfig,build_micro_zoo
from behavior2weights.zoo.planning import plan_zoo_config
from behavior2weights.zoo.public import PublicManifestConfig,build_public_manifest
from behavior2weights.zoo.text import TextZooConfig,build_text_zoo
app=typer.Typer(help="Behavior-to-Weights experimental system.",no_args_is_help=True)
zoo_app=typer.Typer(help="Create and inspect target-model zoos.")
traces_app=typer.Typer(help="Collect black-box behavior traces.")
probes_app=typer.Typer(help="Create candidate query banks.")
train_app=typer.Typer(help="Train inverse models.")
data_app=typer.Typer(help="Prepare public and synthetic datasets.")
stats_app=typer.Typer(help="Power and inferential statistics utilities.")
eval_app=typer.Typer(help="Run locked target-level evaluations.")
analytic_app=typer.Typer(help="Run analytic identifiability experiments.")
app.add_typer(zoo_app,name="zoo")
app.add_typer(traces_app,name="traces")
app.add_typer(probes_app,name="probes")
app.add_typer(train_app,name="train")
app.add_typer(data_app,name="data")
app.add_typer(stats_app,name="stats")
app.add_typer(eval_app,name="evaluate")
app.add_typer(analytic_app,name="analytic")
def _training_tracker(tracking_config:Path|None,output:Path,*,experiment:str,run_name:str,)->ExperimentTracker|None:
    if tracking_config is None:
        config=TrackingConfig(backend="jsonl",repo=str(output/"tracking"),experiment=experiment,enabled=True,)
    else:
        raw=load_yaml(tracking_config)
        config=TrackingConfig.model_validate(raw.get("tracking",raw))
    return create_tracker(config,run_name=run_name)
@analytic_app.command("softmax")
def analytic_softmax(config:Annotated[Path,typer.Option(exists=True,readable=True)],output:Annotated[Path,typer.Option()],)->None:
    raw=load_yaml(config)
    summary=run_linear_softmax_experiment(LinearSoftmaxExperimentConfig.from_dict(raw.get("analytic",raw)),output)
    typer.echo(json.dumps({"systems":summary["systems"],"cells":summary["cells"],"result_path":summary["result_path"],},indent=2,sort_keys=True,))
@app.command()
def doctor()->None:
    info=runtime_info_dict()
    optional={}
    for package in("aim","datasets","transformers","boto3","pyarrow"):
        try:
            module=__import__(package)
            optional[package]=getattr(module,"__version__","installed")
        except ImportError:
            optional[package]=None
    info["optional_dependencies"]=optional
    typer.echo(json.dumps(info,indent=2,sort_keys=True))
@app.command()
def smoke(output:Annotated[Path,typer.Option(help="Output directory.")]=Path("artifacts/smoke"),overwrite:Annotated[bool,typer.Option(help="Remove previous output.")]=False,)->None:
    report=run_smoke_pipeline(output,overwrite=overwrite)
    typer.echo(json.dumps(report,indent=2,sort_keys=True))
@zoo_app.command("plan")
def plan_zoo(config:Annotated[Path,typer.Option(exists=True,readable=True)],kind:Annotated[str,typer.Option(help="auto, micro, text, or public")]="auto",output:Annotated[Path|None,typer.Option()]=None,)->None:
    if kind not in{"auto","micro","text","public"}:
        raise typer.BadParameter("kind must be auto, micro, text, or public")
    projection=plan_zoo_config(load_yaml(config),kind=kind).as_dict()
    rendered=json.dumps(projection,indent=2,sort_keys=True)+"\n"
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(rendered)
    typer.echo(rendered,nl=False)
@zoo_app.command("build-micro")
def build_micro(config:Annotated[Path,typer.Option(exists=True,readable=True)],output:Annotated[Path,typer.Option()],device:Annotated[str,typer.Option()]="cpu",resume:Annotated[bool,typer.Option()]=True,)->None:
    raw=load_yaml(config)
    zoo_config=MicroZooConfig.from_dict(raw.get("zoo",raw))
    policy=SplitPolicy(**raw.get("split_policy",{}))
    result=build_micro_zoo(zoo_config,output,device=device,split_policy=policy,resume=resume)
    typer.echo(json.dumps(manifest_summary(result.records),indent=2,sort_keys=True))
    typer.echo(str(result.manifest_path))
@zoo_app.command("build-text")
def build_text(config:Annotated[Path,typer.Option(exists=True,readable=True)],output:Annotated[Path,typer.Option()],dataset:Annotated[Path|None,typer.Option(exists=True)]=None,device:Annotated[str,typer.Option()]="cuda",)->None:
    raw=load_yaml(config)
    dataset_directory=dataset or Path(raw["dataset_directory"])
    policy=SplitPolicy(**raw.get("split_policy",{}))
    result=build_text_zoo(TextZooConfig.from_dict(raw.get("zoo",raw)),dataset_directory,output,device=device,split_policy=policy,)
    typer.echo(json.dumps(manifest_summary(result.records),indent=2,sort_keys=True))
    typer.echo(str(result.manifest_path))
@zoo_app.command("build-public")
def build_public(config:Annotated[Path,typer.Option(exists=True,readable=True)],output:Annotated[Path,typer.Option()],)->None:
    raw=load_yaml(config)
    records=build_public_manifest(PublicManifestConfig.from_dict(raw),output)
    typer.echo(json.dumps(manifest_summary(records),indent=2,sort_keys=True))
    typer.echo(str(output))
@zoo_app.command("summary")
def zoo_summary(manifest:Annotated[Path,typer.Argument(exists=True)])->None:
    typer.echo(json.dumps(manifest_summary(load_manifest(manifest)),indent=2,sort_keys=True))
@probes_app.command("build")
def build_probes(config:Annotated[Path,typer.Option(exists=True)],output:Annotated[Path,typer.Option()],)->None:
    raw=load_yaml(config)
    bank_config=QueryBankConfig(**raw.get("query_bank",raw))
    include=tuple(raw.get("include",["random","natural"]))
    records=build_query_bank(bank_config,include=include)
    save_query_bank(records,output)
    typer.echo(f"wrote {len(records)} queries to {output}")
@probes_app.command("build-hf")
def build_hf_probes(prompts:Annotated[Path,typer.Option(exists=True,readable=True)],output:Annotated[Path,typer.Option()],tokenizer:Annotated[str,typer.Option(help="Pinned tokenizer name or local path.")],revision:Annotated[str|None,typer.Option()]=None,sequence_length:Annotated[int,typer.Option(min=2)]=128,text_field:Annotated[str,typer.Option()]="text",local_files_only:Annotated[bool,typer.Option()]=False,)->None:
    records=build_hf_query_bank(prompts,output,tokenizer_name=tokenizer,revision=revision,sequence_length=sequence_length,text_field=text_field,local_files_only=local_files_only,)
    typer.echo(f"wrote {len(records)} tokenized queries to {output}")
@traces_app.command("collect-micro")
def collect_micro_traces(manifest:Annotated[Path,typer.Option(exists=True)],queries:Annotated[Path,typer.Option(exists=True)],output:Annotated[Path,typer.Option()],channel:Annotated[ObservationChannel,typer.Option()]=ObservationChannel.LOGITS,feature_dim:Annotated[int|None,typer.Option(min=1)]=None,topk:Annotated[int,typer.Option(min=1)]=8,sample_count:Annotated[int,typer.Option(min=1)]=32,temperature:Annotated[float,typer.Option(min=0.000001)]=1.0,sketch_dim:Annotated[int,typer.Option(min=1)]=16,sketch_seed:Annotated[int,typer.Option()]=17,center_logits:Annotated[bool,typer.Option()]=True,batch_size:Annotated[int,typer.Option(min=1)]=128,base_seed:Annotated[int,typer.Option()]=0,device:Annotated[str,typer.Option()]="cpu",)->None:
    records=load_manifest(manifest,resolve_paths=False)
    query_records=[QueryRecord.model_validate(row)for row in read_jsonl(queries)]
    vocab_sizes={int(record.metadata["model_config"]["vocab_size"])for record in records}
    if len(vocab_sizes)!=1:
        raise typer.BadParameter("selected targets must share one vocabulary size")
    vocab_size=vocab_sizes.pop()
    default_feature_dim=(sketch_dim if channel==ObservationChannel.LOGIT_SKETCH else(2*min(topk,vocab_size)if channel==ObservationChannel.TOPK else vocab_size))
    observation=ObservationConfig(channel=channel,vocab_size=vocab_size,feature_dim=feature_dim or default_feature_dim,topk=min(topk,vocab_size),sample_count=sample_count,temperature=temperature,sketch_dim=sketch_dim,sketch_seed=sketch_seed,center_logits=center_logits,)
    adapter=MicroTransformerAdapter(manifest_root=manifest.parent)
    bundle=collect_target_traces(records,query_records,adapter.load,observation,CollectionConfig(batch_size=batch_size,device=device,base_seed=base_seed),output_directory=output,)
    typer.echo(f"stored traces with shape {tuple(bundle.observations.shape)} in {output}")
@traces_app.command("collect-micro-suite")
def collect_micro_trace_suite(manifest:Annotated[Path,typer.Option(exists=True)],queries:Annotated[Path,typer.Option(exists=True)],experiment:Annotated[Path,typer.Option(exists=True,readable=True)],output:Annotated[Path,typer.Option()],batch_size:Annotated[int,typer.Option(min=1)]=128,base_seed:Annotated[int,typer.Option()]=0,device:Annotated[str,typer.Option()]="cpu",)->None:
    records=load_manifest(manifest,resolve_paths=False)
    query_records=[QueryRecord.model_validate(row)for row in read_jsonl(queries)]
    vocab_sizes={int(record.metadata["model_config"]["vocab_size"])for record in records}
    if len(vocab_sizes)!=1:
        raise typer.BadParameter("selected targets must share one vocabulary size")
    vocab_size=vocab_sizes.pop()
    raw=load_yaml(experiment)
    channel_specs=raw.get("channels")
    if not isinstance(channel_specs,list)or not channel_specs:
        raise typer.BadParameter("experiment config must declare a non-empty channels list")
    adapter=MicroTransformerAdapter(manifest_root=manifest.parent)
    suite_rows:list[dict[str,object]]=[]
    for raw_spec in channel_specs:
        spec={"name":raw_spec}if isinstance(raw_spec,str)else dict(raw_spec)
        observation=ObservationConfig.from_dict(spec,vocab_size=vocab_size)
        channel_output=output/observation.channel.value
        bundle=collect_target_traces(records,query_records,adapter.load,observation,CollectionConfig(batch_size=batch_size,device=device,base_seed=base_seed),output_directory=channel_output,)
        suite_rows.append({"channel":observation.channel.value,"directory":str(channel_output),"shape":list(bundle.observations.shape),"observation_config":dataclasses.asdict(observation),"auxiliary_keys":sorted(bundle.auxiliary),})
    output.mkdir(parents=True,exist_ok=True)
    suite_manifest={"schema_version":1,"experiment":str(experiment),"manifest":str(manifest),"queries":str(queries),"channels":suite_rows,}
    (output/"suite_manifest.json").write_text(json.dumps(suite_manifest,indent=2,sort_keys=True)+"\n")
    typer.echo(json.dumps(suite_manifest,indent=2,sort_keys=True))
@traces_app.command("collect-hf")
def collect_hf_traces(manifest:Annotated[Path,typer.Option(exists=True)],queries:Annotated[Path,typer.Option(exists=True)],output:Annotated[Path,typer.Option()],external_family:Annotated[str|None,typer.Option()]=None,architecture_id:Annotated[str|None,typer.Option()]=None,channel:Annotated[ObservationChannel,typer.Option()]=ObservationChannel.LOGITS,vocab_size:Annotated[int,typer.Option(min=1)]=50_257,feature_dim:Annotated[int|None,typer.Option(min=1)]=None,topk:Annotated[int,typer.Option(min=1)]=32,sample_count:Annotated[int,typer.Option(min=1)]=64,temperature:Annotated[float,typer.Option(min=0.000001)]=1.0,sketch_dim:Annotated[int,typer.Option(min=1)]=512,sketch_seed:Annotated[int,typer.Option()]=17,center_logits:Annotated[bool,typer.Option()]=True,base_seed:Annotated[int,typer.Option()]=0,batch_size:Annotated[int,typer.Option(min=1)]=8,device:Annotated[str,typer.Option()]="cuda",local_files_only:Annotated[bool,typer.Option()]=False,fail_fast:Annotated[bool,typer.Option()]=True,)->None:
    records=load_manifest(manifest,resolve_paths=False)
    records=[record for record in records if(external_family is None or record.factors.get("external_family")==external_family)and(architecture_id is None or record.architecture_id==architecture_id)]
    if not records:
        raise typer.BadParameter("filters selected no targets")
    tokenizer_names={str(record.metadata.get("tokenizer_name"))for record in records}
    if len(tokenizer_names)!=1:
        raise typer.BadParameter("selected targets use multiple tokenizers; collect each tokenizer family separately")
    query_records=[QueryRecord.model_validate(row)for row in read_jsonl(queries)]
    adapter=HuggingFaceCausalLMAdapter(manifest_root=manifest.parent,local_files_only=local_files_only)
    bundle=collect_target_traces(records,query_records,adapter.load,ObservationConfig(channel=channel,vocab_size=vocab_size,feature_dim=feature_dim or(sketch_dim if channel==ObservationChannel.LOGIT_SKETCH else(2*min(topk,vocab_size)if channel==ObservationChannel.TOPK else(min(vocab_size,512)if channel in{ObservationChannel.TOKENS,ObservationChannel.SAMPLE_HISTOGRAM}else vocab_size))),topk=min(topk,vocab_size),sample_count=sample_count,temperature=temperature,sketch_dim=sketch_dim,sketch_seed=sketch_seed,center_logits=center_logits,),CollectionConfig(batch_size=batch_size,device=device,base_seed=base_seed,fail_fast=fail_fast),output_directory=output,)
    typer.echo(f"stored traces with shape {tuple(bundle.observations.shape)} in {output}")
@train_app.command("inverse")
def train_inverse(manifest:Annotated[Path,typer.Option(exists=True)],traces:Annotated[Path,typer.Option(exists=True)],config:Annotated[Path,typer.Option(exists=True)],output:Annotated[Path,typer.Option()],architecture_id:Annotated[str|None,typer.Option()]=None,tracking_config:Annotated[Path|None,typer.Option("--tracking-config",exists=True)]=None,run_name:Annotated[str,typer.Option()]="inverse",seed:Annotated[int|None,typer.Option()]=None,device:Annotated[str|None,typer.Option()]=None,)->None:
    raw=load_yaml(config)
    records=load_manifest(manifest,resolve_paths=False)
    bundle=load_trace_bundle(traces)
    corpus=InverseTrainingCorpus(records,bundle,architecture_id=architecture_id,manifest_root=manifest.parent,canonicalize=bool(raw.get("canonicalize",True)),)
    model_config=InverseModelConfig.from_dict(raw["model"])
    training_config=InverseTrainingConfig(**raw["training"])
    if seed is not None:
        training_config=dataclasses.replace(training_config,seed=seed)
    if device is not None:
        training_config=dataclasses.replace(training_config,device=device)
    tracker=_training_tracker(tracking_config,output,experiment="behavior-to-weights/inverse",run_name=run_name)
    result=train_inverse_model(BehaviorToWeights(model_config),corpus,training_config,output,tracker=tracker)
    typer.echo(json.dumps(dataclasses.asdict(result),indent=2,default=str))
@train_app.command("properties")
def train_properties(manifest:Annotated[Path,typer.Option(exists=True)],traces:Annotated[Path,typer.Option(exists=True)],config:Annotated[Path,typer.Option(exists=True)],output:Annotated[Path,typer.Option()],tracking_config:Annotated[Path|None,typer.Option("--tracking-config",exists=True)]=None,run_name:Annotated[str,typer.Option()]="properties",seed:Annotated[int|None,typer.Option()]=None,device:Annotated[str|None,typer.Option()]=None,)->None:
    raw=load_yaml(config)
    records=load_manifest(manifest,resolve_paths=False)
    bundle=load_trace_bundle(traces)
    training=PropertyTrainingConfig.from_dict(raw["training"])
    if seed is not None:
        training=dataclasses.replace(training,seed=seed)
    if device is not None:
        training=dataclasses.replace(training,device=device)
    corpus=TracePropertyCorpus(records,bundle,properties=training.properties)
    encoder_raw=dict(raw["model"])
    encoder_raw.pop("property_dims",None)
    encoder=InverseModelConfig.from_dict({**encoder_raw,"property_dims":{}})
    tracker=_training_tracker(tracking_config,output,experiment="behavior-to-weights/properties",run_name=run_name)
    result=train_property_classifier(PropertyModelConfig(encoder=encoder,property_dims=corpus.vocabulary.dimensions),corpus,training,output,tracker=tracker,)
    typer.echo(json.dumps(dataclasses.asdict(result),indent=2,default=str))
@data_app.command("train-tokenizer")
def train_tokenizer(config:Annotated[Path,typer.Option(exists=True)],output:Annotated[Path,typer.Option()],overwrite:Annotated[bool,typer.Option()]=False,)->None:
    raw=load_yaml(config)
    tokenizer_config=TokenizerTrainingConfig.from_dict(raw.get("tokenizer",raw))
    typer.echo(str(train_byte_bpe_tokenizer(tokenizer_config,output,overwrite=overwrite)))
@data_app.command("prepare-hf")
def prepare_hf(output:Annotated[Path,typer.Option()],preset:Annotated[str|None,typer.Option()]=None,config:Annotated[Path|None,typer.Option(exists=True)]=None,overwrite:Annotated[bool,typer.Option()]=False,)->None:
    if(preset is None)==(config is None):
        raise typer.BadParameter("provide exactly one of --preset or --config")
    if preset:
        if preset not in PRESETS:
            raise typer.BadParameter(f"unknown preset; choose from {sorted(PRESETS)}")
        dataset_config=PRESETS[preset]
    else:
        assert config is not None
        raw=load_yaml(config)
        dataset_config=HFTextDatasetConfig(**raw.get("dataset",raw))
    typer.echo(str(prepare_hf_causal_lm_dataset(dataset_config,output,overwrite=overwrite)))
@data_app.command("download-transformer-nfn")
def download_nfn(name:Annotated[str,typer.Option(help="mnist or ag_news")],output:Annotated[Path,typer.Option()],expected_sha256:Annotated[str|None,typer.Option()]=None,)->None:
    result=download_transformer_nfn_zoo(name,output,expected_sha256=expected_sha256)
    typer.echo(json.dumps(dataclasses.asdict(result),indent=2,default=str))
@eval_app.command("micro")
def evaluate_micro(manifest:Annotated[Path,typer.Option(exists=True)],traces:Annotated[Path,typer.Option(exists=True)],checkpoint:Annotated[Path,typer.Option(exists=True)],output:Annotated[Path,typer.Option()],architecture_id:Annotated[str|None,typer.Option()]=None,budgets:Annotated[str,typer.Option(help="Comma-separated query budgets.")]="8,16,32,64",query_policies:Annotated[str,typer.Option(help="Comma-separated random,population_disagreement policies.")]="random,population_disagreement",splits:Annotated[str,typer.Option(help="Comma-separated test,ood splits.")]="test,ood",functional_examples:Annotated[int,typer.Option(min=1)]=256,tier:Annotated[str,typer.Option()]="tier1",run_id:Annotated[str,typer.Option()]="evaluation",replicate:Annotated[int,typer.Option(min=0)]=0,seed:Annotated[int,typer.Option()]=20260621,device:Annotated[str,typer.Option()]="cpu",)->None:
    parsed_budgets=tuple(int(value)for value in budgets.split(",")if value.strip())
    parsed_policies=tuple(value.strip()for value in query_policies.split(",")if value.strip())
    parsed_splits=tuple(Split(value.strip())for value in splits.split(",")if value.strip())
    rows=evaluate_micro_inverse(manifest_path=manifest,traces_directory=traces,checkpoint_directory=checkpoint,output_path=output,architecture_id=architecture_id,config=EvaluationConfig(query_budgets=parsed_budgets,query_policies=parsed_policies,splits=parsed_splits,functional_examples=functional_examples,tier=tier,run_id=run_id,replicate=replicate,seed=seed,),device=device,)
    typer.echo(f"wrote {len(rows)} target-level result rows to {output}")
@eval_app.command("properties")
def evaluate_properties(manifest:Annotated[Path,typer.Option(exists=True)],traces:Annotated[Path,typer.Option(exists=True)],checkpoint:Annotated[Path,typer.Option(exists=True)],output:Annotated[Path,typer.Option()],budgets:Annotated[str,typer.Option()]="16,32,64",query_policies:Annotated[str,typer.Option()]="random,population_disagreement",splits:Annotated[str,typer.Option()]="test,ood",tier:Annotated[str,typer.Option()]="tier2",run_id:Annotated[str,typer.Option()]="property-evaluation",replicate:Annotated[int,typer.Option(min=0)]=0,seed:Annotated[int,typer.Option()]=20260621,device:Annotated[str,typer.Option()]="cpu",)->None:
    parsed_budgets=tuple(int(value)for value in budgets.split(",")if value.strip())
    parsed_policies=tuple(value.strip()for value in query_policies.split(",")if value.strip())
    parsed_splits=tuple(Split(value.strip())for value in splits.split(",")if value.strip())
    rows=evaluate_property_classifier(manifest_path=manifest,traces_directory=traces,checkpoint_directory=checkpoint,output_path=output,query_budgets=parsed_budgets,query_policies=parsed_policies,splits=parsed_splits,tier=tier,run_id=run_id,replicate=replicate,seed=seed,device=device,)
    typer.echo(f"wrote {len(rows)} property result rows to {output}")
@stats_app.command("power")
def power(effect:Annotated[float,typer.Option(help="Standardized paired effect size.")],icc:Annotated[float,typer.Option()]=0.5,checkpoints:Annotated[int,typer.Option()]=1,lineages:Annotated[int,typer.Option()]=80,simulations:Annotated[int,typer.Option()]=5_000,)->None:
    result=simulate_clustered_power(lineages=lineages,checkpoints_per_lineage=checkpoints,standardized_effect=effect,intraclass_correlation=icc,simulations=simulations,)
    output=dataclasses.asdict(result)
    output["independent_paired_approximation"]=analytic_paired_sample_size(effect)
    typer.echo(json.dumps(output,indent=2,sort_keys=True))
if __name__=="__main__":
    app()
