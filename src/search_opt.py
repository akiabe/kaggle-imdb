import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
import lightgbm as lgb
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

from functools import partial

import optuna

def optimize(trial, x, y):
    num_leaves = trial.suggest_int("num_leaves", 2, 16)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 10)
    max_depth = trial.suggest_int("max_depth", 3, 9)

    model = lgb.LGBMClassifier(
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        max_depth=max_depth
    )

    kf = model_selection.KFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1.0 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("../input/imdb_train.csv")

    tfidf_vec = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
    )
    tfidf_vec.fit(df.review.values)

    xtrain = tfidf_vec.transform(df.review.values)
    ytrain = df.sentiment.values

    optimization_function = partial(optimize, x=xtrain, y=ytrain)

    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function, n_trials=15)