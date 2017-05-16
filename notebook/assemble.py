import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import skew, mode
from sklearn.preprocessing import LabelEncoder, Normalizer, Imputer
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import json
from Visualization import Display


class DataStream(object):
    """
    This class consists of all functions for the data load and save and prepare for the learning.
    For data load:
    It will read in the training and test data from paths specified in JSON file or de-serializing a Python
    object structure by "unpickling".
    If loading from the raw training and test data file, inside JSON file it has a key-value pair
    as following:
     "paths":
           {"train" : "path/to/training data file",
           "test" : "path/to/test data file"},
    If loading from a Python object structure, inside JSON file it has a key-value pair as following:
      "some unpickling name": "path/to/a Python object structure",

    For data saving:
    """
    @staticmethod
    def load_data(filename):
        """
        Given JSON file including the paths to training and test set, return two raw data
        :param filename: a valid JSON file which contains the training and test data paths
        :return: raw training data set, test data set
        """
        if not os.path.exists(filename):
            raise ValueError("File " + filename + " does not exist.")

        if filename[-4:] != "json":
            raise ValueError(filename + " is not a valid JSON file that contains paths to training and test file.")

        with open(filename) as data_file:
            data = json.load(data_file)

        train_path = data["paths"]["train"]
        train = DataStream.data_read(train_path, "csv")
        test_path = data["paths"]["test"]
        test = DataStream.data_read(test_path, "csv")
        return train, test


    @staticmethod
    def read_for_learn(filename, unpickling):
        """
        Given JSON file including the path to the Python Data Structure, read in train_X, test_X and Y for scikit learn
        model learning
        :param filename: a valid JSON file which contains the path to the Python Data Structure
        :param unpickling: The key to the path
        :return: train_X, test_X and Y
        """

        if not os.path.exists(filename):
            raise ValueError("File " + filename + " does not exist.")

        if filename[-4:] != "json":
            raise ValueError(filename + " is not a valid JSON file that contains paths to training and test file.")

        with open(filename) as data_file:
            data = json.load(data_file)
        try:
            with open(data[unpickling], 'rb') as f:
                data_dict = pickle.load(f)
                return data_dict["train_X"], data_dict["test_X"], data_dict["y"]
        except Exception as e:
            print("Unable to load the data from", data[unpickling], ":", e)

    @staticmethod
    def data_read(path, filetype):
        """
        Load the data into a Pandas DataFrame, for csv file only
        :param path: The full path for the file to load data in
        :param filetype: The type of the file, currently only for csv file
        :return: A pandas DataFrame
        TODO: load data from different file types
        """
        length = -len(filetype)
        if not os.path.exists(path):
            raise ValueError("File " + path + " does not exists.")
        if path[length:] != filetype:
            raise ValueError("File " + path + " needs to be a type of " + filetype)
        return pd.read_csv(path)

    @staticmethod
    def save_dataset(filename, file_key, json_file_name, train_X, test_X, Y):
        """
        Save the training, test and all data as a dictionary into a Python object structure
        :param filename: file name to save as a Python object structure
        :param file_key: the key to add in the JSON file
        :param json_file: the file name of JSON
        :param train: training data
        :param test: test data
        :param all_data: all data with the common columns from training and test data
        :return: None
        """
        total = {"train_X": train_X, "test_X": test_X, "y": Y}
        try:
            filename = "../data/modified/" + filename
            with open(filename, "wb") as f:
                pickle.dump(total, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print("Unable to save data to ", filename, ":", e)

        try:
            with open(json_file_name) as json_file:
                json_decode = json.load(json_file)

            json_decode[file_key] = filename

            with open(json_file_name, "w") as json_file:
                json.dump(json_decode, json_file)
        except Exception as e:
            print("Unable to add new file path to ", json_file_name, ":", e)

    @staticmethod
    def union_datasets(train, test):
        """
        Combine the training and test data with common columns if exists
        :param train: training data set as Pandas DataFrame
        :param test: test data set as Pandas DataFrame
        :return: a Pandas DataFrame combine training and test data with common columns
        """
        if not isinstance(train, pd.DataFrame):
            raise TypeError("Training data is not a valid Pandas DataFrame.")
        if not isinstance(test, pd.DataFrame):
            raise TypeError("Test data is not valid Pandas DataFrame.")

        train_cols = train.columns.values
        test_cols = test.columns.values
        common_cols = list(set(train_cols).intersection(set(test_cols)))
        result = None
        if common_cols:
            print("A new DataFrame is generated combining training and test data...")
            print("The common columns are: ")
            print(common_cols)
            result = pd.concat((train.loc[:, common_cols],
                                test.loc[:, common_cols]), ignore_index=True)
        else:
            print("There is no common column in training and test data.")
        return result

    @staticmethod
    def prepare_for_learn(train, all_data, target):
        """
        Given the training data set and all data set and the predict variable, return the training and test data for
        machine learning model.
        :param train: training data
        :param all_data: the combine of the training and test data without the prediction column
        :param target: the column name of the predict variable
        :return: X_train -- the features of training data,  Y -- prediction column X_test -- features of test data
        """
        if not isinstance(train, pd.DataFrame) or not isinstance(all_data, pd.DataFrame):
            raise TypeError("The training data and the all data both should be Pandas DataFrame")

        if not isinstance(target, str):
            raise TypeError("The name of the target column need to be a string")


        X_train = all_data[:train.shape[0]]
        X_test = all_data[train.shape[0]:]
        y = train[target]

        return X_train, X_test, y


class DataDescribe(object):
    """
    This class will generally describe the input data, including statistical summaries, correlations between variables
    feature importance etc. It is used for the training data.
    """
    def __init__(self, input):
        if not isinstance(input, pd.DataFrame):
            raise TypeError("Input data need to be Pandas DataFrame")
        self.data = input
        self.numerical = None
        self.categorical = None

    @staticmethod
    def calculate_empty(input_data, value=0):
        """
        Find the all the columns with empty values with descending order, where the number of the empty values is larger
        than value
        :param input_data: the input data as a Pandas DataFrame
        :param value: the minimum number of the empty values in the column
        :return: A Pandas Series having empty values with column name and number of empty values
        """
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("Input data is not a valid Pandas DataFrame")
        columns = input_data.isnull().sum()
        columns = columns[columns > value]
        columns.sort_values(ascending=False, inplace=True)
        return columns


    @staticmethod
    def check_types(data):
        for item in data.columns:
            if data.dtypes[item] == "object":
                data[item] = data[item].astype("category")

    def find_empty(self, value=0):
        """
        Show the features with missing values in descending order, in which the number of missing values is larger than
        value
        :param value: the number of missing values
        :return: Nothing
        """
        missing = DataDescribe.calculate_empty(self.data, value)
        if not missing.empty:
            print("The features below have missing values:")
            print(list(missing.index))
            print(missing.plot.bar())

    def find_column_types(self, exclude=None):
        columns = list(self.data.columns)
        if exclude and isinstance(exclude, list):
            for e in exclude:
                if e in columns:
                    columns.remove(e)
            print("Excluding the columns as following:", exclude)

        self.numerical = [f for f in columns if self.data.dtypes[f] != "object"]
        print("The numerical features are", self.numerical)
        self.categorical = [f for f in columns if self.data.dtypes[f] == "object"]
        print("The categorical features are:", self.categorical)
        for item in self.categorical:
            self.data[item] = self.data[item].astype('category')

    def summary(self):
        """
        Print out the general summary on the whole data
        :return: None
        """
        print("The number of observations and the number of features are in (rows, columns) format: ", self.data.shape)
        print("General summary for each feature in the data: ")
        print(self.data.info())
        print("The summary of the data types in the data:")
        print(self.data.get_dtype_counts())
        print("The summary statistics excluding NaN values:")
        print(self.data.describe())

    def show_numerical(self, exclude=None):
        if self.numerical == None:
            self.find_column_types(exclude)
        Display.individual_variable(self.data, self.numerical,sns.distplot)

    def show_category(self, exclude=None, response = None):
        if self.categorical == None:
            self.find_column_types(exclude)
        for item in self.categorical:
            if self.data[item].isnull().any():
                self.data[item] = self.data[item].cat.add_categories(["MISSING"])
                self.data[item] = self.data[item].fillna("MISSING")
        Display.individual_variable(self.data, self.categorical, Display.boxplot, response)

    def show_influence(self, exclude=None, response=None):
        if self.categorical == None:
            self.find_column_types(exclude)

        print("Here is a quick estimation of influence of categorical variables on {0}".format(response))
        a = anova_test(self.data, self.categorical, response)
        a["Disparity"] = np.log(1./a['pval'].values)
        sns.barplot(data=a, x="features", y="Disparity")
        x = plt.xticks(rotation=90)

    def correlations(self, target, threshold=0.5):
        """
        Compute the correlations between variables in the data, list the features and the correlations with target
        variables, and return the features pairs with correlations higher than threshold
        :param target: the column name of the target variable
        :param threshold: the threshold of the correlations between features
        :return: the correlations with the target variable and the feature paris with correlation larger than threshold
        """
        if isinstance(target, str):
            corr = self.data.corr()[target]
            corr = corr[np.argsort(corr, axis=0)[::-1]]
            print(corr)
        else:
            raise TypeError("The column name of the target column needs to be string.")

        features = self.data.corr().drop(target, inplace=True).drop(target, axis=1, inplace=True)
        related_features = (features[abs(features) > threshold][features != 1.0]).unstack().dropna().to_dict()
        values = list(set([(tuple(sorted(key)), related_features[key]) for key in related_features]))
        values = sorted(values, key=lambda x: x[1], reversed=True)
        corr_pairs = pd.DataFrame(values, columns=["Feature Pair", "Correlation"])
        return corr, corr_pairs

class Category(object):
    def __init__(self, input):
        self.data = input
        self.categories = [f for f in self.data.columns if self.data.dtypes[f] == "object"]
        self.encoded = []

    def check_type(self):
        DataDescribe.check_types(self.data)

    def transform(self, train, target, weight=False):
        if isinstance(train, pd.DataFrame) and target in train.columns:
            for item in self.categories:
                self.encode(train, item, target,weight)
                self.encoded.append(item + "_E")

    def encode(self, train, feature, target, weight=False):
        ordering = pd.DataFrame()
        if isinstance(train, pd.DataFrame) and feature in train.columns:
            ordering['val'] = train[feature].unique()
            ordering.index = ordering.val
            ordering['spmean'] = train[[feature, target]].groupby(feature).mean()[target]
            ordering = ordering.sort_values('spmean')
            ordering['ordering'] = range(1, ordering.shape[0] + 1)
            ordering = ordering['ordering'].to_dict()

            for key,val in ordering.items():
                self.data.loc[self.data[feature] == key, feature+"_E"] = val

class DataCleaner(object):

    def __init__(self, input):
        self.all_data = input

    @staticmethod
    def simple_explore(data):
        """
        :param data: Pandas DataFrame to explore
        This function only explore the columns with empty value
        TODO: explore the correlations between variables
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The passing dataset is not a DataFrame")
        print("\nThe shape of the DataFrame is [" + ','.join(map(str, data.shape)) + "]")

        columns = DataDescribe.calculate_empty(data)
        if not columns.empty:
            print("\nThe list below are the columns that has NaN values:")
            print(columns)

        half = data.shape[0]/2
        columns = DataDescribe.calculate_empty(data, half)
        if not columns.empty:
            print("The columns that have more than half empty values:")
            print(list(columns.index))

    @staticmethod
    def current_empty(input_data):
        """
        Compute the list of the column names with empty values
        :param input_data: the input data as a Pandas DataFrame
        :return: A list of the column names with empty values
        """
        cols = list(DataDescribe.calculate_empty(input_data).index)
        if cols:
            print("Now the column name(s) that have empty values are:")
            print(DataDescribe.calculate_empty(input_data))
        else:
            print("There are no empty values in the data.")
        return cols

    def fill_na_spec(self, spec):
        if not isinstance(spec, dict):
            raise TypeError("A dictionary needs to be passed to specify which columns needs to replace with values.")

        col = []
        for key, val in spec.items():
            if isinstance(val, list):
                for item in val:
                    if item in self.all_data.columns.values:
                        self.all_data[item].fillna(key, inplace=True)
                        print("Column {0} 's empty value has been replaced by {1}".format(item, key))
                        col.append(item)
                    else:
                        print("{0} is not a column in the dataset.".format(item))
            else:
                print("To replace NA values with {0}, a list needs to be specified.".format(key))
        print("\n Now the columns have been updated:")
        print(col)
        DataCleaner.current_empty(self.all_data)

    def fill_na_group(self, group):
        """
        This function will go through the dictionary of group, for each column as key[0] has a value of key[1], it will
        fill those columns' empty values in the corresponding list as the mode of the group in column key[0] with value
        key[1] if it is category, or median of the group in column key[0] with value key[1] if it is a numeric one.
        :param group: A dictionary has the key as the tuple of (col1, val1) and value as a list
        :return:
        """
        if not isinstance(group, dict):
            raise TypeError("A dictionary needs to be passed to specify how to fill empty values with group")

        col = []
        for key, val in group.items():
            if key[0] in self.all_data.columns.values:
                if isinstance(val, list):
                    for item in val:
                        if item in self.all_data.columns.values:
                            value = 0
                            if self.all_data[item].dtypes == "object":
                                value = self.all_data.loc[self.all_data[key[0]] == key[1], item].mode()[0]
                            else:
                                value = self.all_data.loc[self.all_data[key[0]] == key[1], item].median()
                            self.all_data[item].fillna(value, inplace=True)
                            print("Now column {0} 's empty value has been replaced by {1}".format(item, value))
                            col.append(item)

                else:
                    print("To replace {0}, a list needs to be specified.".format(key[1]))

            else:
                print("{0} is not a column in the dataset.".format(key[0]))
        print("\n Now the columns have been updated:")
        print(col)
        DataCleaner.current_empty(self.all_data)

    def fill_na_gen(self):
        left = DataCleaner.current_empty(self.all_data)
        col = []
        for val in left:
            value = 0
            if self.all_data[val].dtypes == "object":
                value = self.all_data[val].mode()[0]
            else:
                value = self.all_data[val].median()
            self.all_data[val].fillna(value, inplace=True)
            print("Now column {0} 's empty value has been replaced by {1}".format(val, value))
            col.append(val)
        print("\n Now the columns have been updated:")
        print(col)
        DataCleaner.current_empty(self.all_data)

    def interpret_value(self, spec):
        if not isinstance(spec, dict):
            raise TypeError("A dictionary needs to be passed to specify the columns to be replaced.")

        self.all_data.replace(spec, inplace=True)

    def normalize_numerical(self, display, exclusion):
        if not isinstance(exclusion, list):
            raise TypeError("The exclusion needs to be a list")

        for col in self.all_data.columns.values:
            if self.all_data[col].dtypes in ["float64", "int64"] and col not in exclusion:
                median = self.all_data[col].median()
                outliers = np.where(is_outlier(self.all_data[col]))
                self.all_data[col].iloc[outliers] = median

                if skew(self.all_data[col]) > 0.75:
                    original = pd.Series(self.all_data[col])
                    self.all_data[col] = np.log1p(self.all_data[col])
                    if display:
                        plt.figure(figsize=(10, 5))
                        plt.subplot(1, 2, 1)
                        sns.distplot(original, color='r', hist_kws={'alpha': 0.8})
                        plt.title("Original data")
                        plt.xlabel(col)

                        plt.subplot(1, 2, 2)
                        sns.distplot(self.all_data[col], color='r', hist_kws={'alpha': 0.8})
                        plt.title("Natural log of data")
                        plt.xlabel("Natural log of " + col)
                        plt.tight_layout()

    def convert_value(self, matching, dropping=False):
        if not isinstance(matching, dict):
            raise TypeError("A dictionary needs to be passed to convert the columns.")

        col = []
        for key, val in matching.items():
            if isinstance(val, list):
                for item in val:
                    if item[0] in self.all_data.columns.values and item[1] in self.all_data.columns.values:
                        self.all_data.loc[self.all_data[item[0]] == key[0], item[1]] = key[1]
                        if dropping:
                            self.all_data.drop(item[0], axis=1, inplace=True)
                        col.append(item[1])
                    else:
                        print("Please check the column names, {0} or {1} may not be a valid column name".format(item[0],item[1]))
            else:
                print("The columns to convert need to be a list.")
        print("\n Now the columns have been updated:")
        print(col)


def anova_test(data, features, response):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data needs to be Pandas DataFrame.")

    if not isinstance(features, list):
        raise TypeError("A list of features is needed.")

    if response not in data.columns.values:
        raise ValueError("{0} is not the feature of the data.".format(response))

    anv = pd.DataFrame()
    anv["features"] = features
    pvals = []
    for item in features:
        samples = []
        for i in data[item].unique():
            value = data[data[item] == i][response].values
            samples.append(value)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False otherwise.
    :param points: A numerical observations by multi-dimensions array of observations
    :param thresh: The modified z-score to use a threshold. Observations with a modified
    z-score (based on the median absolute deviation) greater than this value will be classified as outliers
    :return:
       mask : A numeric observations-length boolean array

    Reference:
    Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis = -1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def simple_numerical(data, threshold = 50):
    """
    If any column contains more than threshold numbers, drop the column, otherwise replace with the median for that
    column, remove outliers using Median Absolute Deviation, take the logarithms if skewness > 0.75, then normalize it
    :param data: a 2D DataFrame containing only numerical observations
    threshold: threshold value to drop the column
    :return:
        clean up the array
    """
    if not isinstance(data, pd.DataFrame) and data.ndim != 2:
        raise ValueError("The passing array is not of right shape")


    if np.sum(data.dtypes != "object")!= data.shape[1]:
        raise ValueError("Not all columns in the array is numerical")

    for col in data.columns.values:
        if np.sum(data[col].isnull()) > threshold:
            data = data.drop(col, axis = 1)
        elif np.sum(data[col].isnull()) > 0:
            median = data[col].median()
            idx = np.where(data[col].isnull())[0]
            data[col].iloc[idx] = median

            outliers = np.where(is_outlier(data[col]))
            data[col].iloc[outliers] = median

            if skew(data[col]) > 0.75:
                data[col] = np.log1p(data[col])

            data[col] = Normalizer().fit_transform(data[col].reshape(1,-1))[0]

def simple_categorical(data, threshold = 50):
    """
    If any column contains more than threshold numbers, drop the column, otherwise replace with mode value
    get dummies of all categorical values
    :param data: a 2D DataFrame containing only categorical variables
    :param threshold: threshold value to drop the column
    :return:
       clean up the array
    """
    if not isinstance(data, pd.DataFrame) and data.ndim != 2:
        raise ValueError("The passing array is not of the right shape")

    if np.sum(data.dtypes == "object") != data.shape[1]:
        raise ValueError("Not all columns in the array is categorical variables")

    features = data.columns.values
    for col in features:
        if np.sum(data[col].isnull()) > threshold:
            data = data.drop(col, axis = 1)
    data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))
    data = pd.get_dummies(data)
    return data

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from itertools import product

class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode

        :param n_splits: the number of splits used in mean encoding

        :param target_type: str, 'regression' or 'classification'

        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new