#!/usr/bin/env python3
"""#640: build ONLY the val store for longbench_repobench_p (Qwen3-1.7B).

The benchmark harness (_load_split_samples) reads tokenized/<model>/val.pkl
only; the 16,755-row train tokenization is training-lane data the harness
pass never reads. Reuses the patched prepare.py's own loaders (dedup'd val
= 500 LongBench eval rows) with an empty train stream and no sanity check.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "datasets", "longbench-datasets",
                                "longbench_repobench_p"))

import prepare as rb  # the patched prepare.py (PREPARE_MODEL default 1.7B)

from tokengate.datasets.prepare import prepare_dataset

if __name__ == "__main__":
    prepare_dataset(
        dataset_name=rb.DATASET, task_name=rb.TASK,
        load_train_fn=lambda: iter([]),  # harness pass reads val.pkl only
        load_val_fn=rb.load_val,
        model=os.environ.get("PREPARE_MODEL", "Qwen/Qwen3-1.7B"),
        run_sanity=False)
    print("val-only store done")
