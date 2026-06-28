from __future__ import annotations

import json
import sys
import types

from behavior2weights.config import TrackingConfig
from behavior2weights.tracking.aim import AimTracker
from behavior2weights.tracking.factory import create_tracker


def test_jsonl_tracker_factory_records_tags_and_metrics(tmp_path) -> None:
    tracker = create_tracker(
        TrackingConfig(
            backend="jsonl",
            repo=str(tmp_path),
            experiment="study",
            tags=["confirmatory", "tier1"],
        ),
        run_name="replicate-0",
    )
    assert tracker is not None
    tracker.set_params({"seed": 7})
    tracker.track(0.25, name="loss", step=3, context={"subset": "train"})
    tracker.close("completed")

    event_file = next(tmp_path.glob("study/*/events.jsonl"))
    events = [json.loads(line) for line in event_file.read_text().splitlines()]
    assert events[0]["tags"] == ["confirmatory", "tier1"]
    assert any(event.get("name") == "loss" and event.get("step") == 3 for event in events)
    assert events[-1]["status"] == "completed"


def test_aim_tracker_uses_documented_run_interface(monkeypatch) -> None:
    class FakeRun:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.values = {}
            self.tags = []
            self.metrics = []
            self.name = None
            self.closed = False

        def __setitem__(self, key, value):
            self.values[key] = value

        def add_tag(self, value):
            self.tags.append(value)

        def track(self, value, **kwargs):
            self.metrics.append((value, kwargs))

        def close(self):
            self.closed = True

    module = types.ModuleType("aim")
    module.Run = FakeRun
    module.Text = lambda value: value
    monkeypatch.setitem(sys.modules, "aim", module)

    tracker = AimTracker(
        repo="aim://tracker:53800",
        experiment="b2w",
        run_name="run-1",
        tags=["remote"],
    )
    tracker.set_params({"seed": 1})
    tracker.track(1.5, name="loss", step=2, context={"subset": "train"})
    tracker.close()

    assert tracker.run.kwargs["repo"] == "aim://tracker:53800"
    assert tracker.run.kwargs["experiment"] == "b2w"
    assert tracker.run.tags == ["remote"]
    assert tracker.run.values["status"] == "completed"
    assert tracker.run.closed
