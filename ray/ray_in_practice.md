# Ray Data
- Ray data is Ray's one-stop-shop solution to parallelize preprocessing for various machine learning datasets. 
- Ray data is mainly concerned with parallel execution and fault tolerance. Ray data has native reading/writing/preprocessing APIs, but, more importantly, integrates with your favourite libraries like PyTorch, Xgboost, Pandas, Dask, etc.
## Modin
- If you want to use Pandas on Ray i.e Pandas with the benefits of parallel execution, out-of-memory handling, fault tolerance, etc, you should use [Modin](https://github.com/modin-project/modin).
- Modin speeds up your regular pandas code with just a single change : replace `import pandas as pd` with `import modin.pandas as pd`. 
- I've seen Modin being able to process Gigabytes more data on a single machine than Pandas (which gives OOM). This is because [Modin spills some data to disk for large datasets](https://modin.readthedocs.io/en/stable/getting_started/why_modin/out_of_core.html).
- Modin does NOT have all the features of Pandas. For example, as of 2024, if you want to work directly with sparse matrices, you're better off with just Pandas + Scipy or with Pyspark. Thus, if you're dealing with large datasets and very limited memory, you should probably use PySpark with SparseVector (+ Parquet for storage). [Supported apis](https://modin.readthedocs.io/en/stable/supported_apis/dataframe_supported.html)
