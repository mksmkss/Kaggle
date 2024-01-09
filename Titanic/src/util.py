import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df_train = pd.read_csv("Titanic/assets/train_1.csv")
df_test = pd.read_csv("Titanic/assets/test_1.csv")


def div_title(name):
    title = name.split(",")[1].split(".")[0].strip()
    # print(title)
    if title == "Mr":
        return 0
    elif title == "Miss":
        return 1
    elif title == "Mrs":
        return 2
    elif title == "Master":
        return 3
    else:
        # print(name)
        return 4


def isFemale(Sex):
    if Sex == "female":
        return 0
    else:
        return 1


def conv_sex():
    # use pd.get_dummies to convert sex in train.csv and test.csv
    df_train["Sex"].map(isFemale)
    df_train.to_csv("Titanic/assets/train.csv", index=False)
    df_test["Sex"].map(isFemale)
    df_test.to_csv("Titanic/assets/test.csv", index=False)


def add_title():
    df_train["Title"] = df_train["Name"].apply(div_title)
    df_train.to_csv("Titanic/assets/train_6.csv", index=False)
    # df_test["Title"] = df_test["Name"].apply(div_title)
    # df_test.to_csv("Titanic/assets/test_2.csv", index=False)


def dropna():
    df = pd.read_csv("Titanic/assets/train_2.csv")
    df = df.dropna(subset=["Age"], axis=0)
    df.to_csv("Titanic/assets/train_2.csv", index=False)


def show_age_fare():
    df = pd.read_csv("Titanic/assets/train_2.csv")
    Age = df["Age"]
    Fare = df["Fare"]
    plt.scatter(Age, Fare)
    plt.show()


def fill_age_with_eage():
    df = pd.read_csv("Titanic/assets/test_6.csv")
    Age = df["Age"]
    eAge = df["eAge"]
    df["mAge"] = 0
    for i in enumerate(Age):
        if math.isnan(i[1]):
            df["Age"][i[0]] = eAge[i[0]]
        else:
            df["Age"][i[0]] = i[1]
    df.to_csv("Titanic/assets/test_7.csv", index=False)


def standardlization():
    df = pd.read_csv("Titanic/assets/test_5.csv")
    std_scaler = StandardScaler()
    scaling_columns = ["Age", "Fare", "Title", "Pclass", "SibSp", "Parch", "isAlone"]
    std_scaler.fit(df[scaling_columns])
    df_std = pd.DataFrame(
        std_scaler.transform(df[scaling_columns]),
        columns=scaling_columns,
        index=df.index,
    )
    # 統計取得
    print(df_std.describe())
    df.update(df_std)
    df.to_csv("Titanic/assets/test_6.csv", index=False)


def isAlone(SibSp, Parch):
    if SibSp == 0 and Parch == 0:
        return 0
    else:
        return 1


def add_isAlone():
    df = pd.read_csv("Titanic/assets/train_3.csv")
    df["isAlone"] = df.apply(lambda x: isAlone(x["SibSp"], x["Parch"]), axis=1)
    df.to_csv("Titanic/assets/test_5.csv", index=False)


def relation_survived_isAlone():
    df = pd.read_csv("Titanic/assets/train_4.csv")
    df = df[["Survived", "isAlone"]]
    df = df.groupby("isAlone").mean()
    print(df)


def describe_df():
    df = pd.read_csv("Titanic/assets/train_5.csv")
    print(df.describe())


def whitening():
    df = pd.read_csv("Titanic/assets/train_2.csv")
    df = df[["Age", "Fare", "Title", "Pclass", "SibSp", "Parch"]]
    pca = PCA(n_components=2, whiten=True)
    pca.fit(df)
    data = pca.transform(df)
    print(data)


def add_isNull():
    df = pd.read_csv("Titanic/assets/train_6.csv")
    df["isNull"] = df["Age"].isnull()
    df.to_csv("Titanic/assets/train_6.csv", index=False)


def pairplot():
    df = pd.read_csv("Titanic/assets/train_6.csv")
    df = df[
        [
            "Age",
            "Fare",
            "Title",
            "Pclass",
            "SibSp",
            "Embarked",
            "Parch",
            "Survived",
            "Sex",
            "isNull",
        ]
    ]
    sns.pairplot(df, hue="Survived")
    plt.show()


def div_embarked(embarked):
    if embarked == "S":
        return 0
    elif embarked == "C":
        return 1
    elif embarked == "Q":
        return 2
    else:
        return 3


def conv_embarked():
    df = pd.read_csv("Titanic/assets/train_6.csv")
    df["Embarked"] = df["Embarked"].apply(div_embarked)
    df.to_csv("Titanic/assets/train_6.csv", index=False)


if __name__ == "__main__":
    # conv_sex()
    # add_title()
    # dropna()
    # fill_age_with_eage()
    # standardlization()
    # add_isAlone()
    # relation_survived_isAlone()
    # describe_df()
    # whitening()
    # add_isNull()
    pairplot()
    # conv_embarked()
