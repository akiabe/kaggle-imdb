import pandas as pd

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
        "num_leaves": [2, 4, 8, 16],
        "min_data_in_leaf": [1, 3, 5, 7],
        "max_depth": [3, 5, 7, 9]
    }

    model = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )
    model.fit(xtrain, ytrain)
    print(f"Best score: {model.best_score_}")

    print("Best parameters set")
    print(model.best_estimator_.get_params())

    """
    Best score: 0.84838
    Best parameters set
   {'boosting_type': 'gbdt', 'class_weight': None,
    'colsample_bytree': 1.0, 'importance_type': 'split',
    'learning_rate': 0.1, 'max_depth': 9, 'min_child_samples': 20,
    'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100,
    'n_jobs': -1, 'num_leaves': 16, 'objective': None, 'random_state': None,
    'reg_alpha': 0.0, 'reg_lambda': 0.0, 'silent': True, 'subsample': 1.0,
    'subsample_for_bin': 200000, 'subsample_freq': 0, 'min_data_in_leaf': 3}
    """