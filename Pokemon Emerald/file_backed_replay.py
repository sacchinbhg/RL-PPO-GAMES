"""Keep large Dreamer replay arrays in temporary files while preserving upstream replay content and sampling."""
import json
import mmap
from pathlib import Path
import tempfile
import threading

import numpy as np


def install(directory, report):
    from embodied.core.chunk import Chunk
    from embodied.core.replay import Replay
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = Path(report)
    original_append, original_load, original_save = Chunk.append, Chunk.load, Chunk.save
    original_restore = Replay.load
    io_slots = threading.BoundedSemaphore(2)

    def allocate(shape, dtype):
        if int(np.prod(shape)) * np.dtype(dtype).itemsize < 65536:
            return np.empty(shape, dtype)
        # The mapping owns the storage after this short-lived file handle closes.
        with tempfile.TemporaryFile(dir=directory) as file:
            return np.memmap(file, mode="w+", dtype=dtype, shape=shape)

    def append(chunk, step):
        if not chunk.data:
            chunk.data = {key: allocate((chunk.size, *value.shape), value.dtype) for key, value in step.items()}
        return original_append(chunk, step)

    def load(cls, filename, error="raise"):
        with io_slots:
            # A missed chunk changes which sequences the agent can sample. Propagate the original read error.
            chunk = original_load(filename, error="raise")
            for key in chunk.data:
                value = chunk.data[key]
                if value.nbytes < 65536:
                    continue
                dest = allocate(value.shape, value.dtype)
                dest[:] = value
                dest.flush()
                if hasattr(dest._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                    dest._mmap.madvise(mmap.MADV_DONTNEED)
                chunk.data[key] = dest
        return chunk

    def save(chunk, *args, **kwargs):
        with io_slots:
            return original_save(chunk, *args, **kwargs)

    def restore(replay, *args, **kwargs):
        before = set(replay.chunks)
        result = original_restore(replay, *args, **kwargs)
        restored = sorted(replay.chunks[key].filename for key in set(replay.chunks) - before)
        if restored:
            report.parent.mkdir(parents=True, exist_ok=True)
            draft = report.with_suffix(".tmp")
            draft.write_text(json.dumps(dict(restored_chunks=len(restored), retained_sequences=len(replay),
                every_selected_chunk_read=True, maximum_parallel_chunk_reads_or_writes=2, filenames=restored), indent=2))
            draft.replace(report)
        return result

    Chunk.append = append
    Chunk.load = classmethod(load)
    Chunk.save = save
    Replay.load = restore
