import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

class SnowballTokenizer:
    def __init__(self):
        self.stemmer = SnowballStemmer("english")

    def __call__(self, words):
        return [self.stemmer.stem(word) for word in word_tokenize(words)]

def run(fold):
    df = pd.read_csv("../input/imdb_train_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfidf_vec = TfidfVectorizer(
        tokenizer=SnowballTokenizer(),
        token_pattern=None
    )
    tfidf_vec.fit(df_train.review.values)

    xtrain = tfidf_vec.transform(df_train.review.values)
    xvalid = tfidf_vec.transform(df_valid.review.values)

    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values

    lr = linear_model.LogisticRegression()
    lr.fit(xtrain, ytrain)
    pred = lr.predict_proba(xvalid)[:, 1]

    auc = metrics.roc_auc_score(yvalid, pred)
    print(f"fold={fold}, auc={auc}")

    df_valid.loc[:, "lr_stem_pred"] = pred

    return df_valid[["id","sentiment", "kfold", "lr_stem_pred"]]

if __name__ == "__main__":
    dfs = []
    for j in range(5):
        temp_df = run(j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df)
    fin_valid_df.to_csv("../model_preds/lr_stem.csv", index=False)