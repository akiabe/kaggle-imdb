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
            df = df.merge(temp_df, on="id", how="left")
    #print(df.columns)
    targets = df.sentiment.values
    #print(targets.shape)

    pred_cols = ["lr_pred", "lr_cnt_pred", "lr_stem_pred"]
    for col in pred_cols:
        auc = metrics.roc_auc_score(targets, df[col].values)
        print(f"{col}, overall_auc={auc}")

    print("average")
    avg_pred = np.mean(df[["lr_pred", "lr_cnt_pred", "lr_stem_pred"]].values, axis=1)
    print(avg_pred.shape)
    print(metrics.roc_auc_score(targets, avg_pred))

    print("weighted average")
    lr_pred = df.lr_pred.values
    lr_cnt_pred = df.lr_cnt_pred.values
    lr_stem_pred =df.lr_stem_pred.values
    #print(lr_pred.shape)
    #print(lr_cnt_pred.shape)
    #print(lr_stem_pred.shape)
    avg_pred = (3 * lr_pred + lr_cnt_pred + lr_stem_pred) / 5
    print(avg_pred.shape)
    print(metrics.roc_auc_score(targets, avg_pred))

    print("rank averaging")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    lr_stem_pred = df.lr_stem_pred.rank().values
    avg_pred = (lr_pred + lr_cnt_pred + lr_stem_pred) / 3
    print(avg_pred.shape)
    print(metrics.roc_auc_score(targets, avg_pred))

    print("weighted rank averaging")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    lr_stem_pred = df.lr_stem_pred.rank().values
    avg_pred = (3 * lr_pred + lr_cnt_pred + lr_stem_pred) / 5
    print(avg_pred.shape)
    print(metrics.roc_auc_score(targets, avg_pred))