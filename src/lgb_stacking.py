import glob
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn import metrics

def run(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred", "lr_cnt_pred", "lr_stem_pred"]].values
    xvalid = valid_df[["lr_pred", "lr_cnt_pred", "lr_stem_pred"]].values

    clf = lgb.LGBMClassifier()
    clf.fit(xtrain, train_df.sentiment.values)
    preds = clf.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"{fold}, {auc}")

    valid_df.loc[:, "lgb_pred"] = preds
    return valid_df

if __name__ == "__main__":
    files = glob.glob("../model_preds/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how="left")
    targets = df.sentiment.values
    pred_cols = ["lr_pred", "lr_cnt_pred", "lr_stem_pred"]

    dfs = []
    for j in range(5):
        temp_df = run(df, j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    #print(fin_valid_df.columns)
    #print(fin_valid_df.shape)
    print(
        metrics.roc_auc_score(
            fin_valid_df.sentiment.values,
            fin_valid_df.lgb_pred.values
        )
    )

    #fin_valid_df.to_csv("../model_preds/lr.csv", index=False)
