import pandas as pd

from nltk.tokenize import word_tokenize
from sklearn import decomposition
from sklearn import ensemble
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

    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(xtrain)

    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xvalid)

    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values

    clf = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(xtrain_svd, ytrain)
    pred = clf.predict_proba(xvalid_svd)[:, 1]

    auc = metrics.roc_auc_score(yvalid, pred)
    print(f"fold={fold}, auc={auc}")

    df_valid.loc[:, "rf_svd_pred"] = pred

    return df_valid[["id","sentiment", "kfold", "rf_svd_pred"]]

if __name__ == "__main__":
    dfs = []
    for j in range(5):
        temp_df = run(j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df)
    fin_valid_df.to_csv("../model_preds/rf_svd.csv", index=False)