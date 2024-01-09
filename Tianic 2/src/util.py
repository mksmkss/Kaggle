import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv(
    "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/train/origin_train.csv"
)
df_test = pd.read_csv(
    "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/test/origin_test.csv"
)


# def isFemale(Sex):
#     if Sex == "female":
#         return 0
#     else:
#         return 1


# def conv_sex():
#     # use pd.get_dummies to convert sex in train.csv and test.csv
#     df_train["Sex"].map(isFemale)
#     df_train.to_csv("Titanic/assets/train.csv", index=False)
#     df_test["Sex"].map(isFemale)
#     df_test.to_csv("Titanic/assets/test.csv", index=False)


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


def add_title():
    for i in ["train", "test"]:
        df = pd.read_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/origin_{i}.csv"
        )
        df["Title"] = df["Name"].apply(div_title)
        df.to_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/add_title_{i}.csv",
            index=False,
        )


# def dropna():
#     df = pd.read_csv("Titanic/assets/train_2.csv")
#     df = df.dropna(subset=["Age"], axis=0)
#     df.to_csv("Titanic/assets/train_2.csv", index=False)


# def show_age_fare():
#     df = pd.read_csv("Titanic/assets/train_2.csv")
#     Age = df["Age"]
#     Fare = df["Fare"]
#     plt.scatter(Age, Fare)
#     plt.show()


# def fill_age_with_eage():
#     df = pd.read_csv("Titanic/assets/test_6.csv")
#     Age = df["Age"]
#     eAge = df["eAge"]
#     df["mAge"] = 0
#     for i in enumerate(Age):
#         if math.isnan(i[1]):
#             df["Age"][i[0]] = eAge[i[0]]
#         else:
#             df["Age"][i[0]] = i[1]
#     df.to_csv("Titanic/assets/test_7.csv", index=False)


def standardlization():
    for i in ["test", "train"]:
        df = pd.read_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/eAge_{i}.csv"
        )
        std_scaler = StandardScaler()
        unScaling_columns = [
            "PassengerId",
            "Name",
            "Cabin",
            "Ticket",
        ]
        if i == "train":
            unScaling_columns.append("Survived")
        new_df = df.drop(unScaling_columns, axis=1)
        std_scaler.fit(new_df)
        df_std = pd.DataFrame(
            std_scaler.transform(new_df),
            columns=new_df.columns,
            index=new_df.index,
        )
        # 統計取得
        print(df_std.describe())
        # 結合する
        df = pd.concat([df[unScaling_columns], df_std], axis=1)
        print(df.head())
        df.to_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/stand_eAge_{i}.csv",
            index=False,
        )


def isAlone(SibSp, Parch):
    if SibSp == 0 and Parch == 0:
        return 0
    else:
        return 1


def add_isAlone():
    for i in ["train", "test"]:
        df = pd.read_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/add_title_{i}.csv"
        )
        df["isAlone"] = df.apply(lambda x: isAlone(x["SibSp"], x["Parch"]), axis=1)
        df.to_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/add_title_{i}.csv",
            index=False,
        )


# def relation_survived_isAlone():
#     df = pd.read_csv("Titanic/assets/train_4.csv")
#     df = df[["Survived", "isAlone"]]
#     df = df.groupby("isAlone").mean()
#     print(df)


def describe_df():
    for i in ["train", "test"]:
        df = pd.read_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/origin_{i}.csv"
        )
        print(df.describe())
        print(df.info())


# def whitening():
#     df = pd.read_csv("Titanic/assets/train_2.csv")
#     df = df[["Age", "Fare", "Title", "Pclass", "SibSp", "Parch"]]
#     pca = PCA(n_components=2, whiten=True)
#     pca.fit(df)
#     data = pca.transform(df)
#     print(data)


# def add_isNull():
#     df = pd.read_csv("Titanic/assets/train_6.csv")
#     df["isNull"] = df["Age"].isnull()
#     df.to_csv("Titanic/assets/train_6.csv", index=False)


def pairplot():
    df = pd.read_csv(
        "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/train/add_title_train.csv"
    )
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
            # "isNull",
        ]
    ]
    sns.pairplot(df, hue="Survived")
    plt.show()


def heatmap():
    path = "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/train/stand_eAge_train.csv"
    file_name = path.split("/")[-1].split(".")[0]
    df = pd.read_csv(path)
    print(path.split("/")[-1].split(".")[0])
    # Name,Ticket,Cabinを削除
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1)
    df = df.dropna(subset=["Age"], axis=0)
    plt.subplots(figsize=(10, 10))
    sns.heatmap(df.corr(), annot=True, square=True)
    # plt.show()
    plt.savefig(
        f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/output/img/hm_{file_name}"
    )


def title_survived():
    df = pd.read_csv(
        "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/train/add_title_train.csv"
    )
    df = df[["Title", "Survived"]]
    sns.barplot(x="Title", y="Survived", data=df)
    # plt.show()
    plt.savefig("/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/output/img/title.png")
    # print(df)


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
    for i in ["train", "test"]:
        df = pd.read_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/add_title_{i}.csv"
        )
        df["Embarked"] = df["Embarked"].apply(div_embarked)
        df.to_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/add_title_{i}.csv",
            index=False,
        )


def sibsp_pclass_age():
    # sibsp =[0,1,2,3,4,5], pclass=[1,2,3]なので，18種類の組み合わせごとに年齢の棒グラフを作成.
    df = pd.read_csv(
        "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/train/ambiguous_age_train.csv"
    )
    df = df[["SibSp", "Pclass", "amAge"]]
    plt.figure(figsize=(20, 20))
    for i in range(18):
        plt.subplot(6, 3, i + 1)
        plt.title(f"SibSp:{i//3}, Pclass:{i%3}")
        df_tmp = df[(df["SibSp"] == i // 3) & (df["Pclass"] == i % 3 + 1)]
        sns.histplot(df_tmp["amAge"], bins=8, kde=True)
    plt.tight_layout()
    plt.savefig(
        "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/output/img/sibsp_pclass_age_4.png"
    )


def parch_sibsp_pclass_age():
    # sibsp =[0,1,2,3,4,5], pclass=[1,2,3]，parch=[0,1,2,3,4,5,6]なので，126種類の組み合わせごとに年齢の棒グラフを作成.
    df = pd.read_csv(
        "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/train/ambiguous_age_train.csv"
    )
    df = df[["Parch", "SibSp", "Pclass", "amAge"]]
    plt.figure(figsize=(20, 20))
    for i in range(126):
        plt.subplot(18, 7, i + 1)
        plt.title(f"Parch:{i//21}, SibSp:{i//7%3}, Pclass:{i%7}")
        df_tmp = df[
            (df["Parch"] == i // 21)
            & (df["SibSp"] == i // 7 % 3)
            & (df["Pclass"] == i % 7)
        ]
        sns.histplot(df_tmp["amAge"], bins=8, kde=True)
    plt.tight_layout()
    plt.savefig(
        "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/output/img/parch_sibsp_pclass_age.png"
    )


def conv_age(age):
    return age // 10


def ambiguous_age():
    for i in ["train", "test"]:
        # 10歳ごとに層別
        df = pd.read_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/add_title_{i}.csv"
        )
        # df = df.dropna(subset=["Age"], axis=0)
        df["amAge"] = df["Age"].apply(conv_age)
        df.to_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/ambiguous_age_{i}.csv",
            index=False,
        )


def labelEncode():
    for i in ["train", "test"]:
        df = pd.read_csv(
            f"/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/{i}/add_title_{i}.csv"
        )


def sex_age_survived():
    # 性別と年齢ごとの生存者数を棒グラフで表示
    df = pd.read_csv(
        "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/assets/train/ambiguous_age_train.csv"
    )
    plt.figure(figsize=(20, 20))
    ageRange = 9
    for i in range(2 * ageRange):
        plt.subplot(4, 5, i + 1)
        plt.title(f"age:{(i)//2%ageRange},sex:{i%2}")
        print(f"age:{(i)//2%ageRange},sex:{i%2}")
        df_tmp = df[(df["amAge"] == (i) // 2 % ageRange) & (df["Sex"] == i % 2)]
        sns.histplot(df_tmp["Survived"], bins=2)
    plt.tight_layout()
    plt.savefig(
        "/Users/masataka/Coding/Pythons/Kaggle/Tianic 2/output/img/sex_age_survive"
    )


if __name__ == "__main__":
    # conv_sex()
    # dropna()
    # fill_age_with_eage()
    # relation_survived_isAlone()
    # describe_df()
    # whitening()
    # add_isNull()
    # pairplot()
    # heatmap()
    # title_survived()
    # sibsp_pclass_age()
    # parch_sibsp_pclass_age()
    sex_age_survived()

    # add_title()
    # add_isAlone()
    # conv_embarked()
    # ambiguous_age()
    # standardlization()
