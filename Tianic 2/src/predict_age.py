import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


df_train = pd.read_csv(
    "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/train/ambiguous_age_train.csv"
)
df_test = pd.read_csv(
    "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/test/ambiguous_age_test.csv"
)

# Ageがない場合，削除する
df_train = df_train.dropna(subset=["Age"], axis=0)


X_train, X_test, Y_train, Y_test = train_test_split(
    # df_trainのPclass,SibSp, ParChを利用
    df_train[["Pclass", "SibSp", "Parch"]],  # 説明変数
    # df_train.drop(["amAge", "Name", "Ticket", "Cabin", "Age"], axis=1),
    df_train["amAge"],  # 目的変数
    test_size=0.2,
    random_state=1,
)  # 80%のデータを学習データに、20%を検証データにする

# モデルの作成
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)
model.fit(X_train, Y_train)

# # 　モデルの作成
# model = DecisionTreeClassifier(random_state=1)
# model.fit(X_train, Y_train)

# 結果の確認
Y_pred = model.predict(X_test)
print("accuracy = ", accuracy_score(Y_test, Y_pred))
print("confusion matrix = 正解ラベルが列名，予測ラベルが行名")
print(confusion_matrix(y_true=Y_test, y_pred=Y_pred))

for i in ["test", "train"]:
    miss_index = 0
    df = pd.read_csv(
        f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/ambiguous_age_{i}.csv"
    )
    df["eAge"] = 0
    # Ageが欠損していたらeAgeを代入する
    for j in enumerate(df["Age"]):
        if np.isnan(j[1]):
            df.loc[j[0], "eAge"] = model.predict(df[["Pclass", "SibSp", "Parch"]])[j[0]]
            # print(df["Name"][j[0]])
            miss_index += 1
        else:
            df.loc[j[0], "eAge"] = df.loc[j[0], "amAge"]
    print(miss_index, i)
    df.to_csv(
        f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/eAge_{i}.csv",
        index=False,
    )
