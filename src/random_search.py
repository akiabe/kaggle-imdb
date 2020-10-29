import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
import lightgbm as lgb
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    df = pd.read_csv("../input/imdb_train.csv")

    tfidf_vec = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
    )
    tfidf_vec.fit(df.review.values)

    xtrain = tfidf_vec.transform(df.review.values)
    ytrain = df.sentiment.values

    clf = lgb.LGBMClassifier()
    param_grid = {
        "num_leaves": np.arange(2, 16),
        "min_data_in_leaf": np.arange(1, 10),
        "max_depth": np.arange(3, 9)
    }

    model = model_selection.RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )
    model.fit(xtrain, ytrain)
    print(f"Best score: {model.best_score_}")

    print("Best parameters set")
    print(model.best_estimator_.get_params())