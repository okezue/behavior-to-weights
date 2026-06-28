from __future__ import annotations
from pathlib import Path
from behavior2weights.schemas import Split,TargetRecord
from behavior2weights.train.property import PropertyVocabulary
def make_record(target:str,architecture:str,split:Split)->TargetRecord:
    return TargetRecord(target_id=target,family_id="micro-transformer",lineage_id=target,architecture_id=architecture,task_id="markov",dataset_id="synthetic",seed=0,checkpoint_path=Path("unused"),split=split,)
def test_property_vocabulary_maps_unseen_to_unknown()->None:
    train=[make_record("a","arch-a",Split.TRAIN),make_record("b","arch-b",Split.TRAIN)]
    vocabulary=PropertyVocabulary.fit(train,("architecture_id","task_id"))
    unseen=make_record("c","arch-c",Split.OOD)
    assert vocabulary.encode(unseen,"architecture_id")==0
    assert vocabulary.decode("architecture_id",0)=="__UNKNOWN__"
def test_property_training_config_normalizes_lists()->None:
    from behavior2weights.train.property import PropertyTrainingConfig
    config=PropertyTrainingConfig.from_dict({"properties":["architecture_id","task_id"],"query_budgets":[2,4],"steps":2,"batch_size":1,"validation_every":1,"validation_batches":1,"early_stopping_patience":1,})
    assert config.properties==("architecture_id","task_id")
    assert config.query_budgets==(2,4)
