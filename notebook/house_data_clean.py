import pandas as pd
import numpy as np
import feature
import json

SubClass_mapping = { 20: "1S-New",
                     30: "1S-Old",
                     40: "1S-Attic",
                     45: "1.5S-Finish",
                     50: "1.5S-Unfinished",
                     60: "2S-New",
                     70: "2S-Old",
                     75: "2.5S-All",
                     80: "Split-Level",
                     85: "Split-Foyer",
                     90: "Duplex",
                     120: "1S-PUD-New",
                     150: "1.5S-PUD-All",
                     160: "2S-PUD-New",
                     180: "Split-Level-PUD",
                     190: "2F-All"}


eval_mapping = { "Ex": 5,
                 "Gd": 4,
                 "TA": 3,
                 "Fa": 2,
                 "Po": 1}

group = {
                  ("GarageType", "Detchd"): ["GarageArea", "GarageCars"]
               }

value_matching = {
    ("NoBsmt", 0): [("BsmtFinType1", "BsmtFinSF1"), ("BsmtFinType2", "BsmtFinSF2"), ("BsmtQual", "TotalBsmtSF"),
                    ("BsmtQual", "BsmtUnfSF")
                    ],
    ("None", 0):[("MasVnrType","MasVnrArea")]

}


fill_cols = ["MSZoning", "BsmtUnfSF", "Electrical","KitchenQual",
             "BsmtFinSF1", "Utilities", "SaleType", "Functional", "Exterior1st", "Exterior2nd"]

def main():
    # read in the train and test data
    with open("data.json") as data_file:
        data = json.load(data_file)
    train = pd.read_csv(data["paths"]["train"])
    test = pd.read_csv(data["paths"]["test"])

    print("Looking for the null value in the training dataset...")
    feature.simple_explore(train)

    print("Looking for the null value")
    feature.simple_explore(test)

    cleaner = feature.DataCleaner(train, test)
    cleaner.fill_na_spec(data["fill_dict"])
    cleaner.fill_na_group(group)
    cleaner.fill_na_gen()
    cleaner.current_na()

def cleanup(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The input data is not a valid DataFrame")

    data['MSZoning'].fillna(data.MSZoning.mode(), inplace=True)
    data["Alley"].fillna("NoAc", inplace=True)


    data["Style"] = data["MSSubClass"].map(SubClass_mapping)
    data = data.drop(["MSSubClass"], aixs=1)


    data["MasVnrType"].fillna("None", inplace=True)

    data.loc[data.MasVnrType=="None", "MasVnrArea"] = 0


if __name__ == "__main__": main()