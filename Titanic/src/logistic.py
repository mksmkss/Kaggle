import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score

df_train = pd.read_csv("Titanic/assets/train_4.csv")
df_test = pd.read_csv("Titanic/assets/test_5.csv")
# df_dummy_sex = pd.get_dummies(df_train["Sex"])
# df_train["Sex"] = df_dummy_sex["male"]
# df_dummy_sex = pd.get_dummies(df_test["Sex"])
# df_test["Sex"] = df_dummy_sex["male"]

X_label = ["Pclass", "Sex", "Title", "Age", "isAlone"]
Y_Label = ["Survived"]

X = df_train[X_label]
Y = df_train[Y_Label]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)  # 80%のデータを学習データに、20%を検証データにする

lr = LogisticRegression()  # ロジスティック回帰モデルのインスタンスを作成
lr.fit(X_train, Y_train)  # ロジスティック回帰モデルの重みを学習
# 結果を出力
print("傾き = ", lr.coef_[0][0])
print("切片 = ", lr.intercept_[0])

Y_score = lr.predict_proba(X_test)[:, 1]  # 検証データがクラス1に属する確率
Y_pred = lr.predict(X_test)  # 検証データのクラスを予測
fpr, tpr, thresholds = roc_curve(y_true=Y_test, y_score=Y_score)
print("accuracy = ", accuracy_score(Y_test, Y_pred))
print("confusion matrix = \n", confusion_matrix(y_true=Y_test, y_pred=Y_pred))

# testデータの予測
X_test = df_test[X_label]

df_test["Survived"] = lr.predict(X_test).astype(int)
df_result = df_test[["PassengerId", "Survived"]]
df_result.to_csv("Titanic/assets/result_5.csv", index=False)
