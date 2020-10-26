import glob
import numpy as np
import pandas as pd

from sklearn import metrics

if __name__ == "__main__":
    files = glob.glob("../model_preds/*.csv")
    df = None

    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on=)