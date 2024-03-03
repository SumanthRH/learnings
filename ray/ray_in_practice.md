# Ray Data
- Ray data is Ray's one-stop-shop solution to parallelize preprocessing for various machine learning datasets. 
- Ray data is mainly concerned with parallel execution and fault tolerance. Ray data has native reading/writing/preprocessing APIs, but, more importantly, integrates with your favourite libraries like PyTorch, Xgboost, Pandas, Dask, etc.
## Modin
- If you want to use Pandas on Ray i.e Pnadas with the benefits of parallel execution, out-of-memory handling, fault tolerance, etc, you should use [Modin](https://github.com/modin-project/modin).
- Modin speeds up your regular pandas code with just a single change : replace `import pandas as pd` with `import modin.pandas as pd`. 

