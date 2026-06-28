from __future__ import annotations
import dataclasses
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import cast
import torch
from torch import Tensor
from behavior2weights.data.synthetic import SyntheticDatasetConfig,SyntheticTask,generatesequences
from behavior2weights.schemas import QueryRecord
from behavior2weights.utils import writejsonl
@dataclasses.dataclass(frozen=True,slots=True)
class QueryBankConfig:
    vocab_size:int=32
    seq_len:int=8
    random_queries:int=512
    natural_queries:int=512
    exhaustive_max_vocab:int=8
    exhaustive_max_length:int=5
    seed:int=0
def randomquerytensor(config:QueryBankConfig)->Tensor:
    generator=torch.Generator().manual_seed(config.seed)
    return torch.randint(config.vocab_size,(config.random_queries,config.seq_len),generator=generator,)
def naturalquerytensor(config:QueryBankConfig,task:str="mixture")->Tensor:
    synthetic=SyntheticDatasetConfig(task=cast(SyntheticTask,task),vocab_size=config.vocab_size,seq_len=config.seq_len+1,train_examples=config.natural_queries,validation_examples=1,test_examples=1,seed=config.seed+101,)
    return generatesequences(synthetic,config.natural_queries)[:,:-1]
def exhaustivequerytensor(config:QueryBankConfig)->Tensor:
    if config.vocab_size>config.exhaustive_max_vocab:
        raise ValueError("exhaustive query enumeration disabled: vocab_size exceeds exhaustive_max_vocab")
    if config.seq_len>config.exhaustive_max_length:
        raise ValueError("exhaustive query enumeration disabled: seq_len exceeds exhaustive_max_length")
    values=list(itertools.product(range(config.vocab_size),repeat=config.seq_len))
    return torch.tensor(values,dtype=torch.long)
def torecords(tensor:Tensor,*,source:str,partition:str="candidate")->list[QueryRecord]:
    records:list[QueryRecord]=[]
    for index,row in enumerate(tensor.tolist()):
        records.append(QueryRecord(query_id=f"{source}-{index:08d}",input_ids=row,source=source,partition=partition,))
    return records
def buildquerybank(config:QueryBankConfig,*,include:Sequence[str]=("random","natural"),)->list[QueryRecord]:
    records:list[QueryRecord]=[]
    if "random" in include:
        records.extend(torecords(randomquerytensor(config),source="random"))
    if "natural" in include:
        records.extend(torecords(naturalquerytensor(config),source="natural"))
    if "exhaustive" in include:
        records.extend(torecords(exhaustivequerytensor(config),source="exhaustive"))
    return records
def savequerybank(records:Sequence[QueryRecord],path:str|Path)->None:
    writejsonl(path,[record.model_dump(mode="json")for record in records])
