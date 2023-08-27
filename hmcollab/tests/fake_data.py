import numpy as np
import pandas as pd


def customer_id(i):
    return f"0{i:x}"


def fake_customers(n, r=None):
    if r is None:
        r = np.random.RandomState(42)
    ids = [customer_id(i) for i in range(n)]
    ages = r.randint(100, size=n)
    return pd.DataFrame.from_dict({"customer_id": ids, "age": ages})
