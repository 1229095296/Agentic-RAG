"""Minimal Slime stubs for unit-testing the release without a training install."""

from __future__ import annotations

import sys
import types
from enum import Enum


try:
    import slime  # noqa: F401
except ImportError:
    slime = types.ModuleType("slime")
    slime.__path__ = []
    rollout = types.ModuleType("slime.rollout")
    rollout.__path__ = []
    sglang_rollout = types.ModuleType("slime.rollout.sglang_rollout")
    data_source = types.ModuleType("slime.rollout.data_source")
    utils = types.ModuleType("slime.utils")
    utils.__path__ = []
    http_utils = types.ModuleType("slime.utils.http_utils")
    type_utils = types.ModuleType("slime.utils.types")

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        ABORTED = "aborted"

    class Sample:
        Status = Status

        def __init__(self, prompt="", response="", label=None, **kwargs):
            self.prompt = prompt
            self.response = response
            self.label = label
            self.metadata = kwargs.pop("metadata", {})
            self.status = kwargs.pop("status", Status.PENDING)
            for key, value in kwargs.items():
                setattr(self, key, value)

    class GenerateState:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("GenerateState requires a real Slime installation")

    class RolloutDataSource:
        pass

    class RolloutDataSourceWithBuffer:
        pass

    async def post(*args, **kwargs):
        raise RuntimeError("HTTP generation requires a real Slime installation")

    sglang_rollout.GenerateState = GenerateState
    data_source.RolloutDataSource = RolloutDataSource
    data_source.RolloutDataSourceWithBuffer = RolloutDataSourceWithBuffer
    http_utils.post = post
    type_utils.Sample = Sample

    sys.modules.update(
        {
            "slime": slime,
            "slime.rollout": rollout,
            "slime.rollout.sglang_rollout": sglang_rollout,
            "slime.rollout.data_source": data_source,
            "slime.utils": utils,
            "slime.utils.http_utils": http_utils,
            "slime.utils.types": type_utils,
        }
    )
