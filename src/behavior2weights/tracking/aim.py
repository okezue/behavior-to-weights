from __future__ import annotations
import importlib
from collections.abc import Iterable
from typing import Any
from behavior2weights.tracking.base import ExperimentTracker
class AimTracker(ExperimentTracker):
    def __init__(self,*,repo:str,experiment:str,run_name:str|None=None,tags:Iterable[str]=(),system_tracking_interval:float|None=10.0,)->None:
        try:
            aim_module=importlib.import_module("aim")
            run_class=aim_module.Run
        except(ImportError,AttributeError)as error:
            raise RuntimeError("Aim 3.x is not installed. Install behavior2weights[aim], or select backend=jsonl.")from error
        self.run=run_class(repo=repo,experiment=experiment,system_tracking_interval=system_tracking_interval,)
        if run_name:
            self.run.name=run_name
        for tag in tags:
            addtag=getattr(self.run,"add_tag",None)
            if callable(addtag):
                addtag(str(tag))
            else:
                self.run["tags"]=list(tags)
                break
    def setparams(self,params:dict[str,Any])->None:
        self.run["params"]=params
    def track(self,value:float,*,name:str,step:int,context:dict[str,Any]|None=None,)->None:
        self.run.track(float(value),name=name,step=step,context=context or{})
    def logtext(self,text:str,*,name:str,step:int=0)->None:
        try:
            aim_module=importlib.import_module("aim")
            text_class=aim_module.Text
            self.run.track(text_class(text),name=name,step=step)
        except(ImportError,AttributeError):
            self.run[f"text/{name}/{step}"]=text
    def close(self,status:str="completed")->None:
        try:
            self.run["status"]=status
        finally:
            close=getattr(self.run,"close",None)
            if callable(close):
                close()
