from typing import Iterable

import concurrent
import datasets
import glob
import json
import multiprocessing
import os
import tqdm
import ray
from ray.experimental.tqdm_ray import tqdm

ray.init() # uses all cpus by default

tqdm_remote = ray.remote(tqdm)

@ray.remote
def load_mmaped_table(filepath, tqdm_ref):
    table = datasets.table.MemoryMappedTable.from_file(filepath)
    tqdm_ref.update.remote(1)
    return table

def load_dataset_tables(
    files: Iterable[str]
) -> Iterable[datasets.table.MemoryMappedTable]:
    
    tqdm_ref = tqdm_remote.remote(total=len(files), desc=f"Loading {len(files)}")
    result = ray.get([load_mmaped_table.remote(file, tqdm_ref) for file in files])

    return result


def datasets_fast_load_from_disk(cache_path: str) -> datasets.Dataset:
    print(f"fast_load_from_disk called with path:", cache_path)
    dataset_info_path = os.path.join(cache_path, "dataset_info.json")
    with open(dataset_info_path, encoding="utf-8") as dataset_info_file:
        dataset_info = datasets.DatasetInfo.from_dict(json.load(dataset_info_file))

    # dataset_state_path = os.path.join(cache_path, "state.json")
    # with open(dataset_state_path, encoding="utf-8") as state_file:
    #     state = json.load(state_file)

    files = glob.glob(os.path.join(cache_path, "*.arrow"))
    files = sorted(files)
    ds_tables = load_dataset_tables(
        files=files,
    )
    arrow_table = datasets.table.concat_tables(ds_tables)

    # split = state["_split"]
    split = None
    split = datasets.splits.Split(split) if split is not None else split
    try:
        dataset = datasets.Dataset(
            arrow_table=arrow_table,
            info=dataset_info,
            split=split,
            # fingerprint=state["_fingerprint"],
        )
    except ValueError as e:
        dataset = datasets.Dataset(
            arrow_table=arrow_table,
            split=split,
            # fingerprint=state["_fingerprint"],
        )
    return dataset

if __name__ == "__main__":
    import time
    start = time.time()
    dataset = datasets_fast_load_from_disk(cache_path="/Users/sumanthrh/.cache/huggingface/datasets/glue/wnli/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad")
    end = time.time()
    