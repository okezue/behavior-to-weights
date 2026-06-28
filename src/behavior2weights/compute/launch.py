from __future__ import annotations
import dataclasses
import shlex
from pathlib import Path
@dataclasses.dataclass(frozen=True,slots=True)
class SlurmSpec:
    job_name:str
    command:tuple[str,...]
    partition:str="gpu"
    time:str="24:00:00"
    nodes:int=1
    gpus_per_node:int=1
    cpus_per_task:int=8
    memory:str="64G"
    array:str|None=None
    account:str|None=None
    constraint:str|None=None
    environment_setup:tuple[str,...]=()
def renderslurm(spec:SlurmSpec)->str:
    directives=["#!/usr/bin/env bash","set -euo pipefail",f"#SBATCH --job-name={spec.job_name}",f"#SBATCH --partition={spec.partition}",f"#SBATCH --time={spec.time}",f"#SBATCH --nodes={spec.nodes}",f"#SBATCH --gpus-per-node={spec.gpus_per_node}",f"#SBATCH --cpus-per-task={spec.cpus_per_task}",f"#SBATCH --mem={spec.memory}",]
    if spec.array:
        directives.append(f"#SBATCH --array={spec.array}")
    if spec.account:
        directives.append(f"#SBATCH --account={spec.account}")
    if spec.constraint:
        directives.append(f"#SBATCH --constraint={spec.constraint}")
    directives.extend(["",*spec.environment_setup,"",shlex.join(spec.command),""])
    return "\n".join(directives)
def writeslurm(spec:SlurmSpec,path:str|Path)->Path:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(renderslurm(spec))
    path.chmod(0o755)
    return path
