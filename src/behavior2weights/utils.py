from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import random
import subprocess
from collections.abc import Iterable,Mapping
from pathlib import Path
from typing import Any,TypeVar,cast
import numpy as np
import torch
import yaml
T=TypeVar("T")
def seedeverything(seed:int,*,deterministic:bool=True)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True,warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
def stablehash(value:Any,*,length:int=16)->str:
    payload=json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]
def filesha256(path:str|Path,chunk_size:int=1024*1024)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb")as handle:
        while chunk:=handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
def atomicwritejson(path:str|Path,value:Any)->None:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=jsondefault)+"\n")
    tmp.replace(path)
def readjson(path:str|Path)->Any:
    return json.loads(Path(path).read_text())
def writejsonl(path:str|Path,rows:Iterable[Mapping[str,Any]])->None:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8")as handle:
        for row in rows:
            handle.write(json.dumps(dict(row),sort_keys=True,default=jsondefault)+"\n")
    tmp.replace(path)
def readjsonl(path:str|Path)->list[dict[str,Any]]:
    with Path(path).open(encoding="utf-8")as handle:
        return[json.loads(line)for line in handle if line.strip()]
def loadyaml(path:str|Path)->dict[str,Any]:
    data=yaml.safe_load(Path(path).read_text())
    if data is None:
        return{}
    if not isinstance(data,dict):
        raise TypeError(f"Expected mapping at {path}, found {type(data).__name__}")
    return data
def dumpyaml(path:str|Path,value:Mapping[str,Any])->None:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(dict(value),sort_keys=False))
def deepmerge(base:Mapping[str,Any],override:Mapping[str,Any])->dict[str,Any]:
    result:dict[str,Any]=dict(base)
    for key,value in override.items():
        if key in result and isinstance(result[key],Mapping)and isinstance(value,Mapping):
            result[key]=deepmerge(result[key],value)
        else:
            result[key]=value
    return result
def gitmetadata(root:str|Path|None=None)->dict[str,Any]:
    root_path=Path(root or ".").resolve()
    def command(*args:str)->str|None:
        try:
            return subprocess.check_output(["git","-C",str(root_path),*args],stderr=subprocess.DEVNULL,text=True).strip()
        except(OSError,subprocess.CalledProcessError):
            return None
    return{"commit":command("rev-parse","HEAD"),"branch":command("rev-parse","--abbrev-ref","HEAD"),"dirty":bool(command("status","--porcelain")),}
def dataclasstodict(value:Any)->dict[str,Any]:
    if not dataclasses.is_dataclass(value):
        raise TypeError(f"Expected dataclass, found {type(value).__name__}")
    return dataclasses.asdict(cast(Any,value))
def chunked(items:list[T],size:int)->Iterable[list[T]]:
    if size<=0:
        raise ValueError("size must be positive")
    for index in range(0,len(items),size):
        yield items[index:index+size]
def resolvepath(path:str|Path,*,root:str|Path|None=None)->Path:
    expanded=Path(os.path.expandvars(os.path.expanduser(str(path))))
    if expanded.is_absolute()or root is None:
        return expanded
    return Path(root)/expanded
def jsondefault(value:Any)->Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(cast(Any,value))
    if isinstance(value,Path):
        return str(value)
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value,set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
