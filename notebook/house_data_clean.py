import pandas as pd
import numpy as np
import feature
import json

interpret_mapping = {
    'MSSubClass': { 20: "SubClass_20",
                     30: "SubClass_30",
                     40: "SubClass_40",
                     45: "SubClass_45",
                     50: "SubClass_50",
                     60: "SubClass_60",
                     70: "SubClass_70",
                     75: "SubClass_75",
                     80: "SubClass_80",
                     85: "SubClass_85",
                     90: "SubClass_90",
                     120: "SubClass_120",
                     150: "SubClass_150",
                     160: "SubClass_160",
                     180: "SubClass_180",
                     190: "SubClass_190"},
    'Street': { 'Grvl' : 0,
                'Pave': 1},
    'Alley': { 'Grvl': 0,
               'Pave': 1,
               'NoAccess': -1},
    'LotShape': {'Reg' : 0,
                 'IR1' : 1,
                 'IR2' : 2,
                 'IR3' : 3 },
    'Utilities': {'AllPub': 3,
                  'NoSewr' : 2,
                  'NoSeWa': 1,
                  'ELO': 0},
    'LandSlope': {'Gtl' : 0,
                  'Mod' : 1,
                  'Sev' : 2},
    'ExterQual': {'Ex' : 5,
                  'Gd' : 4,
                  'TA' : 3,
                  'Fa' : 2,
                  'Po' : 1},
    'ExterCond': {'Ex' : 5,
                  'Gd' : 4,
                  'TA' : 3,
                  'FA' : 2,
                  'Po' : 1},
    'BsmtQual': {'Ex' : 5,
                 'Gd' : 4,
                 'TA' : 3,
                 'Fa' : 2,
                 'Po' : 1,
                 'NoBsmt' : 0},
    'BsmtCond': {'Ex' : 5,
                 'Gd' : 4,
                 'TA' : 3,
                 'Fa' : 2,
                 'Po' : 1,
                 'NoBsmt' : 0},
    'BsmtExposure' : {'Gd' : 3,
                      'Av' : 2,
                      'Mn' : 1,
                      'No' : 0,
                      'NoBsmt' : 0},
    'HeatingQC' :{'Ex' : 5,
                  'Gd' : 4,
                  'TA' : 3,
                  'Fa' : 2,
                  'Po' : 1},
    'CentralAir' :{'N' : 0,
                   'Y' : 1},
    'KitchenQual' :{'Ex' : 5,
                    'Gd' : 4,
                    'TA' : 3,
                    'Fa' : 2,
                    'Po' : 1},
    'Functional' : {'Typ' : 0,
                    'Min1' : 1,
                    'Min2' : 1,
                    'Mod' : 2,
                    'Maj1' : 3,
                    'Maj2' : 3,
                    'Sev' : 4,
                    'Sal' : 5},
    'FireplaceQu' : {'Ex' : 5,
                     'Gd' : 4,
                     'TA' : 3,
                     'Fa' : 2,
                     'Po' : 1,
                     },
    'GarageFinish' : {'Fin' : 2,
                      'RFn' : 1,
                      'Unf' : 0,
                      'NoGarage': 0},
    'GarageQual' : {'Ex' : 5,
                    'Gd' : 4,
                    'TA' : 3,
                    'Fa' : 2,
                    'Po' : 1,
                    'NoGarage': 0},
    'GarageCond' : {'Ex': 5,
                    'Gd' : 4,
                    'TA' : 3,
                    'Fa' : 2,
                    'Po' : 1,
                    'NoGarage' : 0},
    'PavedDrive' : {'Y' : 2,
                    'P' : 1,
                    'N' : 0},
      'PoolQC': {'Ex' : 4,
                 'Gd' : 3,
                 'TA' : 2,
                 'Fa' : 1,
                 'NoPool': 0},
      'Fence': {'GdPrv': 2,
                'MnPrv': 1,
                'GdWo': 2,
                'MnWw': 1,
                'NoFence': 0
                },
}

exclusion = ['Street', 'PravedDrive', 'Fence', 'GarageCond', 'GarageQual', 'GarageFinish', 'Id', 'Alley', 'LotShape',
             'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual','BsmtCond', 'BsmtExposure', 'HeatingQC',
             'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'OverallQual', 'OverallCond','YearBuilt',
             'YearRemodAdd','GarageYrBlt', 'MoSold', 'YrSold']

group = {("GarageType", "Detchd"): ["GarageArea", "GarageCars"]}

value_matching = {
    ("NoBsmt", 0): [("BsmtFinType1", "BsmtFinSF1"), ("BsmtFinType2", "BsmtFinSF2"), ("BsmtQual", "TotalBsmtSF"),
                    ("BsmtQual", "BsmtUnfSF")
                    ],
    ("None", 0):[("MasVnrType","MasVnrArea")],
    ("NoGarage", 0) : [("GarageType", "GarageCars"), ("GarageType", "GarageArea")],
    ("NoPool", 0): [("PoolQC", "PoolArea")],
    ("NoFirePlace", 0): [('FireplaceQu', "Fireplaces")]

}

mapping = {("None", 0): ["MiscFeature", "MiscVal"]}


fill_cols = ["MSZoning", "BsmtUnfSF", "Electrical","KitchenQual",
             "BsmtFinSF1", "Utilities", "SaleType", "Functional", "Exterior1st", "Exterior2nd"]

def main():
    # read in the train and test data
    cleaner = feature.DataCleaner("data.json")
    with open("data.json") as data_file:
        data = json.load(data_file)
    cleaner.fill_na_spec(data["fill_dict"])
    cleaner.fill_na_group(group)
    cleaner.fill_na_gen()
    cleaner.interpret_value(interpret_mapping)

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