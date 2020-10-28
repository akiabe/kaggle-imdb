import pandas as pd

from nltk.tokenize import word_tokenize
import lightgbm as lgb
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

def run(fold):
    df = pd.read_csv("../input/imdb_train_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfidf_vec = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
    )
    tfidf_vec.fit(df_train.review.values)

    xtrain = tfidf_vec.transform(df_train.review.values)
    xvalid = tfidf_vec.transform(df_valid.review.values)

    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values

    clf = lgb.LGBMClassifier()
    clf.fit(xtrain, ytrain)
    pred = clf.predict_proba(xvalid)[:, 1]

    auc = metrics.roc_auc_score(yvalid, pred)
    print(f"fold={fold}, auc={auc}")

    df_valid.loc[:, "lgb_pred"] = pred

    return df_valid[["id", "sentiment", "kfold", "lgb_pred"]]

if __name__ == "__main__":
    dfs = []
    for j in range(5):
        temp_df = run(j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    #fin_valid_df.to_csv("../model_preds/lgb.csv", index=False)