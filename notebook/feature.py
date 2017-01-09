import numpy as np
import pandas as pd
from scipy.stats import skew, mode
from sklearn.preprocessing import LabelEncoder, Normalizer, Imputer
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
def simple_explore(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The passing dataset is not a DataFrame")

    print("The shape of the DataFrame is [" + ','.join(map(str, data.shape)) + "]")

    print("The list below are the columns that has NaN values:")
    columns = data.isnull().sum()
    print(columns[columns > 0])

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

    def __init__(self, train, test):
        if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
            raise TypeError("Train or Test dataset is not a DataFrame")
        self.train = train
        self.test = test
        self.all_data = self._union_datasets()
        print("Now after the training and test data have been combined:")
        simple_explore(self.all_data)
        self.cols = list(self._calculate_null().index)

    def _union_datasets(self):

        train_cols = self.train.columns.values
        test_cols = self.test.columns.values

        common_cols = list(set(train_cols).intersection(set(test_cols)))

        result = None
        if common_cols:
            print("The common columns are: ")
            print(common_cols)
            result = pd.concat((self.train.loc[:, common_cols],
                                self.test.loc[:, common_cols]), ignore_index=True)
        else:
            print("There is no common column in training and test datasets.")

        return result

    def _calculate_null(self):
        columns = self.all_data.isnull().sum()
        return columns[columns > 0]

    def _find_pred(self):
        train_cols = self.train.columns.values
        test_cols = self.test.columns.values
        return list(set(train_cols).difference(set(test_cols)))[0]

    def prep_for_learn(self):
        X = pd.get_dummies(self.all_data, sparse = True)
        X = X.fillna(0)
        X_train = X[:self.train.shape[0]]
        X_test = X[self.train.shape[0]:]
        y = self.train[self._find_pred()]
        return X_train, X_test, y


    def current_na(self):
        if self.cols:
            print("Now the column name(s) that have empty values are:")
            print(self._calculate_null())
        else:
            print("There are no empty values in the dataset.")

    def fill_na_spec(self, spec):
        if not isinstance(spec, dict):
            raise TypeError("A dictionary needs to be passed to specify which columns needs to replace with values.")

        for key, val in spec.items():
            if isinstance(val, list):
                for item in val:
                    if item in self.all_data.columns.values:
                        self.all_data[item].fillna(key, inplace=True)
                        self.cols.remove(item)
                    else:
                        print("{0} is not a column in the dataset.".format(item))
            else:
                print("To replace NA values with {0}, a list needs to be specified.".format(key))
        self.current_na()

    def fill_na_group(self, group):
        if not isinstance(group, dict):
            raise TypeError("A dictionary needs to be passed to specify how to fill empty values with group")

        for key, val in group.items():
            if key[0] in self.all_data.columns.values:
                if isinstance(val, list):
                    for item in val:
                        if item in self.all_data.columns.values:
                            if self.all_data[item].dtypes == "object":
                                value = self.all_data.loc[self.all_data[key[0]] == key[1], item].mode()
                            else:
                                value = self.all_data.loc[self.all_data[key[0]] == key[1], item].median()
                            self.all_data[item].fillna(value, inplace=True)
                            self.cols.remove(item)
                else:
                    print("To replace {0}, a list needs to be specified.".format(key[1]))

            else:
                print("{0} is not a column in the dataset.".format(key[0]))

        self.current_na()

    def fill_na_gen(self):
        if not isinstance(self.cols, (pd.Series, pd.DataFrame, list)):
            raise TypeError("The passing column is not a right data type.")

        left = list(self.cols)
        for val in left:
            if self.all_data[val].dtypes == "object":
                self.all_data[val].fillna(self.all_data[val].mode(), inplace=True)
            else:
                self.all_data[val].fillna(self.all_data[val].median(), inplace=True)
            self.cols.remove(val)
        self.current_na()

    def interpret_value(self, spec):
        if not isinstance(spec, dict):
            raise TypeError("A dictionary needs to be passed to specify the columns to be replaced.")

        self.all_data.replace(spec, inplace=True)

    def normalize_numerical(self, display, exclusion):
        if not isinstance(exclusion, list):
            raise TypeError("The exclusion needs to be a list")

        for col in self.all_data.columns.values:
            if self.all_data[col].dtypes != "object" and col not in exclusion:
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

    def save_learning(self, filename):
        X_train, X_test, y = self.prep_for_learn()
        data = {'X_train': X_train, 'X_test': X_test, 'y': y}
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', filename, ':', e)