from typing import Iterable

import concurrent
import datasets
import glob
import json
import multiprocessing
import os
import tqdm

def load_dataset_tables(
    files: Iterable[str], num_workers: int
) -> Iterable[datasets.table.MemoryMappedTable]:

    use_threads = False
    if use_threads:
        pool_cls = concurrent.futures.ThreadPoolExecutor
        pool_kwargs = {"max_workers": num_workers}
    else:
        pool_cls = multiprocessing.Pool
        pool_kwargs = {"processes": num_workers}
    
    with pool_cls(**pool_kwargs) as pool:
        result = list(
            tqdm.tqdm(
                pool.imap(datasets.table.MemoryMappedTable.from_file, files),
                desc=f"Loading {len(files)} files with {num_workers} workers",
                total=len(files),
            )
        )
    return result


def datasets_fast_load_from_disk(cache_path: str) -> datasets.Dataset:
    print(f"fast_load_from_disk called with path:", cache_path)
    dataset_info_path = os.path.join(cache_path, "dataset_info.json")
    with open(dataset_info_path, encoding="utf-8") as dataset_info_file:
        dataset_info = datasets.DatasetInfo.from_dict(json.load(dataset_info_file))

    dataset_state_path = os.path.join(cache_path, "state.json")
    with open(dataset_state_path, encoding="utf-8") as state_file:
        state = json.load(state_file)

    files = glob.glob(os.path.join(cache_path, "*.parquet"))
    files = sorted(files)
    num_workers = 16
    ds_tables = load_dataset_tables(
        files=files,
        num_workers=num_workers
    )
    arrow_table = datasets.table.concat_tables(ds_tables)

    split = state["_split"]
    split = datasets.splits.Split(split) if split is not None else split

    return datasets.Dataset(
        arrow_table=arrow_table,
        info=dataset_info,
        split=split,
        fingerprint=state["_fingerprint"],
    )

if __name__ == "__main__":
    import time
    start = time.time()
    dataset = datasets_fast_load_from_disk(cache_path="./")