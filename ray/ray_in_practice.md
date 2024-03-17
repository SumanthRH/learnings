# Setting up a Ray Cluster

# Ray Data
- Ray data is Ray's one-stop-shop solution to parallelize preprocessing for various machine learning datasets. 
- Ray data is mainly concerned with parallel execution and fault tolerance. Ray data has native reading/writing/preprocessing APIs, but, more importantly, integrates with your favourite libraries like PyTorch, Xgboost, Pandas, Dask, etc.
## Modin
- If you want to use Pandas on Ray i.e Pandas with the benefits of parallel execution, out-of-memory handling, fault tolerance, etc, you should use [Modin](https://github.com/modin-project/modin).
- Modin speeds up your regular pandas code with just a single change : replace `import pandas as pd` with `import modin.pandas as pd`. 
- I've seen Modin being able to process Gigabytes more data on a single machine than Pandas (which gives OOM). This is because [Modin spills some data to disk for large datasets](https://modin.readthedocs.io/en/stable/getting_started/why_modin/out_of_core.html).
- Modin does NOT have all the features of Pandas. For example, as of 2024, if you want to work directly with sparse matrices, you're better off with just Pandas + Scipy or with Pyspark. Thus, if you're dealing with large datasets and very limited memory, you should probably use PySpark with SparseVector (+ Parquet for storage). [Supported apis](https://modin.readthedocs.io/en/stable/supported_apis/dataframe_supported.html)

## Why Ray Data?
- Overall, you still need to use a data processing library like pandas or pytorch to handle data type specific operations (`merge` or `describe`, etc) while working with Ray Data. This can make it not-so-intuitive as to why we might need Ray Data in the first place.
## A Simple Example: Multi-node batched inference
- One simple example for this is batch inference with Ray Data. Let's say that you trained a PyTorch model. You now want to perform batched inference on a large test dataset. Let's say you have Gigabytes of images and a multi-node cluster. With Ray Data it is pretty straightforward to distribute this across a cluster:

```python
dataset = ray.data.read_images(<image_dir>)

class MyPredictor:
    def __init__(self, model_path):
        ckpt = torch.load(model_path)
        self.model = MyModel()
        self.model.load_state_dict(ckpt)

    def __call__(self, batch: np.array):
        batch_pt = torch.tensor(batch)
        return self.model(batch_pt)
    
dataset = dataset.map_batches(
    MyPredictor, 
    fn_constructor_args=[<model_path>], 
    batch_size=128,
    concurrency=3,
    num_cpus=4
)
```

If you've got 4 CPUs in each node and 3 nodes, then the above code would run batched inference concurrently across 3 nodes. In each node, internally, a Ray actor is instantiated (basically a `MyPredictor` object) that will handle the execution of the `__call__` method for different batches of data. 

You might say that there are similar functionalities in different libraries like PySpark, but Ray's model is general and thus you can use a similar predictor class + map batches call for various types of data - dataframes, images, text, etc.

# Ray Train
Ray Train is a library for distributed ML model training powered by Ray. 

# Ray Tune

# So...Ray for everything?