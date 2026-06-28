from __future__ import annotations
import pytest
from behavior2weights.compute.smoke import run_smoke_pipeline
@pytest.mark.slow
def test_end_to_end_smoke(tmp_path)->None:
    report=run_smoke_pipeline(tmp_path/"smoke",overwrite=True)
    assert report["status"]=="completed"
    assert report["zoo"]["targets"]>0
    assert report["weight_nrmse"]>=0
