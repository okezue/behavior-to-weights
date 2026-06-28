from __future__ import annotations
import pytest
from behavior2weights.compute.smoke import runsmokepipeline
@pytest.mark.slow
def testendtoendsmoke(tmp_path)->None:
    report=runsmokepipeline(tmp_path/"smoke",overwrite=True)
    assert report["status"]=="completed"
    assert report["zoo"]["targets"]>0
    assert report["weight_nrmse"]>=0
