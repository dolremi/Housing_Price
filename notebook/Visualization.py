import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Display:

    @staticmethod
    def individual_variable(data, vars, plot, id=None):
        """
        Plot the individual in a grid fashion
        :param data: the raw data
        :param vars: variables will be plot
        :param plot: plot type
        :param id: id variable will be compared
        :return: None
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The data needs to be Pandas DataFrame")

        if id and isinstance(id, str):
            f = pd.melt(data, id_vars=[id], value_vars=vars)
            g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
            g = g.map(plot, "value", id)
        else:
            f = pd.melt(data, value_vars=vars)
            g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False,size=5)
            g = g.map(plot, "value")

    @staticmethod
    def corr_heatmap(data):
        if not isinstance(data. pd.Dataframe):
            raise TypeError("Input data should be a Pandas DataFrame")

        corr = data.select_dtypes(include = ["float64", "int64"]).corr()
        plt.figure(figsize=(12,12))
        sns.heatmap(corr, vmax=1, square=True)