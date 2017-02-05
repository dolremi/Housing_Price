import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Display:

    @staticmethod
    def corr_heatmap(data):
        if not isinstance(data. pd.Dataframe):
            raise TypeError("Input data should be a Pandas DataFrame")

        corr = data.select_dtypes(include = ["float64", "int64"]).corr()
        plt.figure(figsize=(12,12))
        sns.heatmap(corr, vmax=1, square=True)