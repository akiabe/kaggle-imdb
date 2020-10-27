import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/imdb_train.csv")
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )
    df["kfold"] = -1
    df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, "kfold"] = fold

    df.to_csv("../input/imdb_train_folds.csv", index=True, index_label="id")

    #test = pd.read_csv("../input/imdb_train_folds.csv")
    #print(test.head())