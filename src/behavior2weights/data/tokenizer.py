from __future__ import annotations
import dataclasses
import json
from pathlib import Path
from typing import Any
@dataclasses.dataclass(frozen=True,slots=True)
class TokenizerTrainingConfig:
    dataset_name:str
    dataset_config:str|None=None
    revision:str|None=None
    split:str="train"
    text_column:str="text"
    vocab_size:int=2_048
    min_frequency:int=2
    sample_size:int|None=1_000_000
    batch_size:int=1_000
    special_tokens:tuple[str,...]=("<|pad|>","<|unk|>","<|endoftext|>")
    @classmethod
    def fromdict(cls,raw:dict[str,Any])->TokenizerTrainingConfig:
        data=dict(raw)
        if "special_tokens" in data:
            data["special_tokens"]=tuple(data["special_tokens"])
        return cls(**data)
def trainbytebpetokenizer(config:TokenizerTrainingConfig,output_directory:str|Path,*,overwrite:bool=False,)->Path:
    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer,decoders,models,pre_tokenizers,trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as error:
        raise RuntimeError("Install behavior2weights[hf] to train a tokenizer")from error
    output=Path(output_directory)
    manifest_path=output/"tokenizer_manifest.json"
    if manifest_path.exists()and not overwrite:
        return manifest_path
    dataset=load_dataset(config.dataset_name,config.dataset_config,split=config.split,revision=config.revision,)
    if config.sample_size is not None:
        dataset=dataset.select(range(min(config.sample_size,len(dataset))))
    def iterator()->Any:
        for start in range(0,len(dataset),config.batch_size):
            batch=dataset[start:start+config.batch_size]
            yield[str(value or "")for value in batch[config.text_column]]
    tokenizer=Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=config.vocab_size,min_frequency=config.min_frequency,special_tokens=list(config.special_tokens),show_progress=True,)
    tokenizer.train_from_iterator(iterator(),trainer=trainer,length=len(dataset))
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token="<|pad|>",unk_token="<|unk|>",eos_token="<|endoftext|>",)
    output.mkdir(parents=True,exist_ok=True)
    fast.save_pretrained(output)
    manifest={"schema_version":1,"config":dataclasses.asdict(config),"dataset_fingerprint":dataset._fingerprint,"dataset_rows":len(dataset),"actual_vocab_size":len(fast),"tokenizer_files":sorted(path.name for path in output.iterdir()if path.is_file()),}
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest_path
