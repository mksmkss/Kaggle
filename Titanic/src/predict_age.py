import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv("Titanic/assets/train_5.csv")
df_test = pd.read_csv("Titanic/assets/test_6.csv")

X_label = ["Pclass", "Title", "isAlone"]
X = df_train[X_label]

Y = df_train[["Age"]]

X_dtest = df_test[X_label]


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=0,
)  # 80%のデータを学習データに、20%を検証データにする

# xgboostのオブジェクトに変換する
dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_test, label=Y_test)
dtest = xgb.DMatrix(X_dtest)


# grid_params = {
#     "learning_rate": [0.2, 0.6, 0.8, 1],
#     "max_depth": [3, 6, 9],
#     "reg_lambda": [0.1, 0.5, 1],
# }

# grid = XGBClassifier()

# grid_search = GridSearchCV(
#     grid, param_grid=grid_params, scoring="accuracy", error_score="raise"
# )

# le = LabelEncoder()
# Y_train = le.fit_transform(Y_train)
# Y_train = np.reshape(Y_train, (-1))

# print(len(Y_train))

# grid_search.fit(X_train, Y_train)

# print(grid_search.best_params_)

params = {
    "objective": "reg:squarederror",
    "silent": 1,
    "random_state": 1234,
    # 学習用の指標 (RMSE)
    "eval_metric": "rmse",
}

num_round = 500
watchlist = [(dtrain, "train"), (dvalid, "eval")]  # 訓練データはdtrain、評価用のテストデータはdvalidと設定

model = xgb.train(
    params,
    dtrain,  # 訓練データ
    num_round,  # 設定した学習回数
    early_stopping_rounds=20,
    evals=watchlist,
)

# 予測
prediction_XG = model.predict(dtest, ntree_limit=model.best_ntree_limit)

# 小数を丸めている
# prediction_XG = np.round(prediction_XG)

df_test["eAge"] = prediction_XG
df_test.to_csv("Titanic/assets/test_6.csv", index=False)
print(df_test)
