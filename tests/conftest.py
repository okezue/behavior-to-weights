from __future__ import annotations
import contextlib
import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
if str(SRC)not in sys.path:
    sys.path.insert(0,str(SRC))
@pytest.fixture(autouse=True,scope="session")
def single_threaded_torch()->None:
    torch.set_num_threads(1)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
