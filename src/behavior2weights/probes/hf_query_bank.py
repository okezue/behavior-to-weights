from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from behavior2weights.schemas import QueryRecord
from behavior2weights.utils import read_jsonl,write_jsonl
def build_hf_query_bank(prompts_path:str|Path,output_path:str|Path,*,tokenizer_name:str,revision:str|None=None,sequence_length:int=128,text_field:str="text",local_files_only:bool=False,)->list[QueryRecord]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install behavior2weights[hf] to tokenize public-model probes")from error
    if sequence_length<2:
        raise ValueError("sequence_length must be at least two")
    rows=read_jsonl(prompts_path)
    if not rows:
        raise ValueError("prompt file is empty")
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_name,revision=revision,trust_remote_code=False,local_files_only=local_files_only,)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token=tokenizer.eos_token
    records:list[QueryRecord]=[]
    for index,row in enumerate(rows):
        if text_field not in row:
            raise ValueError(f"row {index} is missing text field {text_field!r}")
        text=str(row[text_field])
        token_ids=tokenizer.encode(text,add_special_tokens=False)
        token_ids=token_ids[-sequence_length:]
        if len(token_ids)<sequence_length:
            token_ids=[int(tokenizer.pad_token_id)]*(sequence_length-len(token_ids))+token_ids
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest()
        metadata:dict[str,Any]=dict(row.get("metadata",{}))
        metadata.update({"prompt_sha256":digest,"tokenizer_name":tokenizer_name,"tokenizer_revision":revision,"original_token_count":len(tokenizer.encode(text,add_special_tokens=False)),"sequence_length":sequence_length,})
        records.append(QueryRecord(query_id=str(row.get("query_id",f"hf-{digest[:16]}-{index:06d}")),input_ids=[int(value)for value in token_ids],source=str(row.get("source","hf-prompt-suite")),partition=str(row.get("partition","candidate")),metadata=metadata,))
    if len({record.query_id for record in records})!=len(records):
        raise ValueError("query_id values must be unique")
    write_jsonl(output_path,[record.model_dump(mode="json")for record in records])
    return records
