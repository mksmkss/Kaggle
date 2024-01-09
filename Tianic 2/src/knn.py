from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv(
    "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/train/stand_eAge_train.csv"
)

df_test = pd.read_csv(
    "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/test/stand_eAge_test.csv"
)

X_train, X_test, Y_train, Y_test = train_test_split(
    df_train[["Title", "Fare", "Sex", "Pclass", "eAge"]],  # 説明変数
    df_train["Survived"],  # 目的変数
    test_size=0.2,
    random_state=1,
)  # 80%のデータを学習データに、20%を検証データにする

# モデルの作成
model = KNeighborsClassifier(n_neighbors=5)
print(X_train.info(), Y_train.info())
model.fit(X_train, Y_train)

# 結果の確認
Y_pred = model.predict(X_test)
print("accuracy = ", accuracy_score(Y_test, Y_pred))
print("confusion matrix = 正解ラベルが列名，予測ラベルが行名")
print(confusion_matrix(y_true=Y_test, y_pred=Y_pred))
