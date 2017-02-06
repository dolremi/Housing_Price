import numpy as np
import pandas as pd
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
    def load_data(filename, unpickling=None):
        """
        Two ways to load data: from JSON file, from a Python object structure
        :param filename: a valid JSON file which contains the training and test data paths
        :param unpickling: the key of de-serializing a valid Python object structure
        :return: raw training data set, test data set and the total data set with common columns from two raw data
        """
        if not os.path.exists(filename):
            raise ValueError("File " + filename + " does not exist.")

        if filename[-4:] != "json":
            raise ValueError(filename + " is not a valid JSON file that contains paths to training and test file.")

        with open(filename) as data_file:
            data = json.load(data_file)
        if unpickling:
            try:
                with open(data[unpickling], 'rb') as f:
                    data_dict = pickle.load(f)
                    return  data_dict["train"], data_dict["test"], data_dict["all"]
            except Exception as e:
                print("Unable to load the data from", data[unpickling], ":", e)
        else:
            train_path = data["paths"]["train"]
            train = DataStream.data_load(train_path, "csv")
            test_path = data["paths"]["test"]
            test = DataStream.data_load(test_path, "csv")
            all_data = DataStream.union_datasets(train, test)
            return train, test, all_data

    @staticmethod
    def data_load(path, filetype):
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
    def save_dataset(filename, file_key, json_file, train, test, all_data):
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
        total = {"train": train, "test": test, "all": all_data}
        try:
            with open(filename, "wb") as f:
                pickle.dump(total, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print("Unable to save data to ", filename, ":", e)

        try:
            with open(json_file) as json_file:
                json_decode = json.load(json_file)

            json_decode[file_key] = filename

            with open(json_file, "w") as json_file:
                json.dump(json_decode, json_file)
        except Exception as e:
            print("Unable to add new file path to ", json_file, ":", e)

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

        if train.shape[1] == all_data.shape[1] + 1:
            X_train = all_data[:train.shape[0]]
            X_test = all_data[train.shape[0]:]
            y = train[target]
        else:
            raise TypeError("The training data and the all data don't match in the shape. ")
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


class DataCleaner(object):

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

        columns = DataCleaner.calculate_null(data)
        if not columns.empty:
            print("\nThe list below are the columns that has NaN values:")
            print(columns)

        half = data.shape[0]/2
        columns = DataCleaner.calculate_null(data, half)
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
        cols = list(DataCleaner.calculate_null(input_data).index)
        if cols:
            print("Now the column name(s) that have empty values are:")
            print(DataCleaner.calculate_null(input_data))
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

    def save_learning(self, filename):
        X_train, X_test, y = self.prep_for_learn()
        data = {'X_train': X_train, 'X_test': X_test, 'y': y}
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to ', filename, ':', e)