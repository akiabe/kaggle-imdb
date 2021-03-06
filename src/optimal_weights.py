import glob
import numpy as np
import pandas as pd

from sklearn import metrics
from functools import partial
from scipy.optimize import fmin

class OptimizeAUC:
    def __init__(self):
        self.coef = 0

    def _auc(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        auc_score = metrics.roc_auc_score(y, predictions)
        return -1.0 * auc_score

    def fit(self, X, y):
        partial_loss = partial(self._auc, X=X, y=y)
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        self.coef_ = fmin(partial_loss, initial_coef, disp=True)

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions

def run(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred", "lr_cnt_pred", "lr_stem_pred"]].values
    xvalid = valid_df[["lr_pred", "lr_cnt_pred", "lr_stem_pred"]].values

    opt = OptimizeAUC()
    opt.fit(xtrain, train_df.sentiment.values)
    preds = opt.predict(xvalid)
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"{fold}, {auc}")

    return opt.coef_

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

    coefs = []
    for j in range(5):
        coefs.append(run(df, j))
    coefs = np.array(coefs)

    print("coefs")
    print(coefs)

    print("mean coefs")
    coefs = np.mean(coefs, axis=0)
    print(coefs)

    print("optimal weighted average")
    wt_avg = (
        coefs[0] * df.lr_pred.values
        + coefs[1] * df.lr_cnt_pred.values
        + coefs[2] * df.lr_stem_pred.values
    )
    print(wt_avg)
    print(metrics.roc_auc_score(targets, wt_avg))
