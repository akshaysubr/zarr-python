from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

import zarr
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.registry import register_pipeline


@dataclass(frozen=True)
class CompactSelectionTestPipeline(BatchedCodecPipeline):
    calls: ClassVar[int] = 0

    async def read_basic_selection(self, *, out, **kwargs: object) -> bool:
        del kwargs
        type(self).calls += 1
        out.as_ndarray_like()[...] = 17
        return True


register_pipeline(CompactSelectionTestPipeline)


@pytest.mark.parametrize("use_explicit_basic", [False, True])
def test_compact_reader_runs_before_chunk_projection_materialization(
    use_explicit_basic: bool,
) -> None:
    path = f"{__name__}.CompactSelectionTestPipeline"
    CompactSelectionTestPipeline.calls = 0
    with zarr.config.set({"codec_pipeline.path": path}):
        array = zarr.create_array({}, shape=(8, 9), chunks=(2, 3), dtype="int32", zarr_format=3)
        result = (
            array.get_basic_selection((slice(None), slice(None)))
            if use_explicit_basic
            else array[:]
        )

    np.testing.assert_array_equal(result, np.full((8, 9), 17, dtype=np.int32))
    assert CompactSelectionTestPipeline.calls == 1
