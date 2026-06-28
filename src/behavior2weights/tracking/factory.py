from __future__ import annotations
from behavior2weights.config import TrackingConfig
from behavior2weights.tracking.aim import AimTracker
from behavior2weights.tracking.base import ExperimentTracker
from behavior2weights.tracking.jsonl import JsonlTracker
def createtracker(config:TrackingConfig,*,run_name:str|None=None)->ExperimentTracker|None:
    if not config.enabled or config.backend=="none":
        return None
    if config.backend=="jsonl":
        return JsonlTracker(config.repo,experiment=config.experiment,run_name=run_name,tags=config.tags,)
    if config.backend=="aim":
        return AimTracker(repo=config.repo,experiment=config.experiment,run_name=run_name,tags=config.tags,system_tracking_interval=config.system_tracking_interval,)
    raise ValueError(f"Unknown tracking backend: {config.backend}")
