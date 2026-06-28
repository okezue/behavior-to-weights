from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Any
from behavior2weights.tracking.base import ExperimentTracker
class JsonlTracker(ExperimentTracker):
    def __init__(self,directory:str|Path,*,experiment:str,run_name:str|None=None,tags:tuple[str,...]|list[str]=(),)->None:
        self.run_id=uuid.uuid4().hex
        self.directory=Path(directory)/experiment/self.run_id
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/"events.jsonl"
        self._handle=self.path.open("a",encoding="utf-8")
        self._write({"type":"run_start","run_id":self.run_id,"experiment":experiment,"run_name":run_name,"tags":list(tags),"time":time.time(),})
    def _write(self,event:dict[str,Any])->None:
        self._handle.write(json.dumps(event,sort_keys=True,default=str)+"\n")
        self._handle.flush()
    def setparams(self,params:dict[str,Any])->None:
        self._write({"type":"params","params":params,"time":time.time()})
    def track(self,value:float,*,name:str,step:int,context:dict[str,Any]|None=None,)->None:
        self._write({"type":"metric","name":name,"value":float(value),"step":int(step),"context":context or{},"time":time.time(),})
    def logtext(self,text:str,*,name:str,step:int=0)->None:
        self._write({"type":"text","name":name,"text":text,"step":step,"time":time.time()})
    def close(self,status:str="completed")->None:
        if self._handle.closed:
            return
        self._write({"type":"run_end","status":status,"time":time.time()})
        self._handle.close()
