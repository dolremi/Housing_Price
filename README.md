 [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-technologies) Kaggle Competition
For Data Analysis:
* [Overall Summary notebook](notebook/Data/General_summary.ipynb)


Currently there are two notebooks:
* features.ipynb do all the data exploration, missing data handling and feature constructions
* Linear model.ipynb now has a simple Lasso model, one can modify the code to load the data and run different linear model.
In the meantime, several py files are implemented for a library for data pipeline.
* In assemble.py, there are several classes including DataStream, DataDescribe can be used for simple explore. And DataCleaner now do the missing data handlers. And Category and Numerical do the data types handling.
* In Visualization.py, there is simple individual variable plot, more functions will be added.
* __Be Caution__, Methods API in those classes are still under constructions, may change without notice, please check the latest code to use them. 