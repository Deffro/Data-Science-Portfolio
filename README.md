# Data Science Portfolio by Dimitrios Effrosynidis

Α compilation of notebooks I created for Data Science related tasks like Tutorials, Exploratory Data Analysis, and Machine Learning.
More notebooks will be added as I learn things and devote time to write about them.

Visit [my website](https://deffro.github.io/) or my [Medium](https://medium.com/@dimitris.effrosynidis) profile, where I include everything listed here and much more.

Below it is a summary of them.

## :movie_camera: Topic Modeling on my Watched Movies

The code is located [here](https://github.com/Deffro/Data-Science-Portfolio/tree/master/Notebooks/Topic%20Modelling%20on%20my%20Watched%20Movies) and the related article [here](https://medium.com/analytics-vidhya/topic-modeling-on-my-watched-movies-1d17491803b4).

1. Use Wikipedia to grab movies and more specifically their Summaries and Plots
2. Merge IMDb data with Wikipedia
3. Build, Evaluate and Visualize an LDA model

## :mag_right: Outlier Detection — Theory, Visualizations, and Code

The [article](https://towardsdatascience.com/outlier-detection-theory-visualizations-and-code-a4fd39de540c) is available on Towards Data Science and the code is located [here](https://github.com/Deffro/Data-Science-Portfolio/blob/master/Notebooks/Outlier%20Detection/Outlier%20Detection%20-%20Theory%2C%20Visualizations%20and%20Code.ipynb).

1. What is Outlier Detection?
2. Causes
3. Applications
4. Approaches
5. Taxonomy
6. Algorithms - Isolation Forest, Extended Isolation Forest, Local Outlier Factor, DBSCAN, One Class SVM, Ensemble

## :fire: Exploratory Data Analysis for the popular Battle Royale game PUBG

This is a very popular [kaggle kernel](https://www.kaggle.com/deffro/eda-is-fun) with more than 1250 upvotes and 80.000 views, with which I won the **1st prize** for the best kernel in that Kaggle competition.

## :clock930: Time Series Analysis with Theory, Plots, and Code

Two articles on Towards Data Science ([Part 1](https://towardsdatascience.com/time-series-analysis-with-theory-plots-and-code-part-1-dd3ea417d8c4), [Part 2](https://towardsdatascience.com/time-series-analysis-with-theory-plots-and-code-part-2-c72b447da634)). Code is available [here](https://github.com/Deffro/Data-Science-Portfolio/tree/master/Notebooks/Time%20Series%20Analysis%20and%20Forecasting).

1. What is a Time Series?
2. The Basic Steps in a Forecasting Task
3. Time Series Graphics (Time Plot, Seasonal Plot, Seasoonal Subseries Plot, Lag Scatter Plot)
4. Time Series Components
5. Stationarity
6. Autocorrelation
7. Moving Average, Double and Triple Exponential Smoothing

## :boom: Forecasting Wars: Classical Forecasting Methods vs Machine Learning

The task is to forecast, as precisely as possible, the unit sales (demand) of various products sold in the USA by Walmart. Competitors: Simple Exponential Smoothing, Double Exponential Smoothing, Triple Exponential Smoothing, ARIMA, SARIMA, SARIMAX, Light Gradient Boosting, Random Forest, Linear Regression.

The [article](https://towardsdatascience.com/forecasting-wars-classical-forecasting-methods-vs-machine-learning-4fd5d2ceb716) is available on Towards Data Science and the code is located [here](https://github.com/Deffro/Data-Science-Portfolio/tree/master/Notebooks/10%2B1%20Cross%20Validation%20Techniques%20Visualized).

## :house_with_garden: Clustering Neighborhoods

[This](https://github.com/Deffro/Data-Science-Portfolio/tree/master/Notebooks/Clustering%20Neighborhouds) is a project that aims to help practicing some technologies and Data Science.

Let's suppose that you live in Toronto, Canada (you can do this for every city that has enough data) and you found a better job. This job is located in the other side of the city and you decide that you need to re-locate closer. You really like your neighborhood though, and you want to find a similar one.

This code uses the venues of each neighborhood as features in a clustering algorithm (k-means) and finds similar neighborhoods.

Things that were used

1. **Beautiful Soup** - Package that lets us extract the content of a web page into simple text
2. **Json** - Handle json files and transform them into a pandas dataframe
3. **Geocode** - Package that converts an address to its coordinates
4. **Scikit Learn** - Machine learning package in order to use clustering
5. **Folium** - Package to create spatial maps. NOTE: Maps that are created from folium are not displayed in jupyter notebook. I provide links to them as static images.


## &#x1F4D9; Pandas Tutorial

Are you starting with Data Science? Pandas is perhaps the first best thing you will need. And it's really easy!

After reading (and practising) [this](https://github.com/Deffro/Data-Science-Portfolio/blob/master/Notebooks/PandasTutorial.ipynb) tutorial you will learn how to:

- Create, add, remove and rename columns
- Read, select and filter data
- Retrieve statistics for data
- Sort and group data
- Manipulate data

## :straight_ruler: Normalization and Standardization

Normalization/standardization are designed to achieve a similar goal, which is to create features that have similar ranges to each other and are widely used in data analysis to help the programmer to get some clue out of the raw data.

[This](https://github.com/Deffro/Data-Science-Portfolio/blob/master/Notebooks/Normalization-Standardization.ipynb) notebook includes:

- Normalization
- Why normalize?
- Standardization
- Why standardization?
- Differences?
- When to use and when not
- Python code for Simple Feature Scaling, Min-Max, Z-score, log1p transformation

## :wrench: Encoding Categorical Features

Python code on how to transform nominal and ordinal variables to integers.

[This](https://github.com/Deffro/Data-Science-Portfolio/blob/master/Notebooks/Encoding%20Categorical%20Features.ipynb) Notebook includes:

- Ordinal Encoding with LabelEncoder, Panda's Factorize, and Panda's Map
- Nominal Encoding with One-Hot Encoding and Binary Encoding

## :bar_chart: Visualizations with Seaborn

Every plot that *seaborn* provides is here with examples in a real dataset.

[This](https://github.com/Deffro/Data-Science-Portfolio/blob/master/Notebooks/Visualizations%20with%20Seaborn.ipynb) notebook includes:
- Theory on Skewness and Kurtosis
- Univariate plots. [Histogram, KDE, Box plot, Count plot, Pie chart]
- Bivariate plots. [Scatter plot, Join plot, Reg plot, KDE plot, Hex plot, Line plot, Bar plot, Violin plot, Boxen plot, Strip plot]
- Multivariate plots. [Correlation Heatmap, Pair plot, Scatter plot, Line plot, Bar plot]

## :clock1030: Feature Engineering with Dates

In [this](https://github.com/Deffro/Data-Science-Portfolio/blob/master/Notebooks/Feature-Engineering-with-Dates.ipynb) tutorial I present the datetime format that Pandas provides to handle datetime features. In the end I create a function that generates 23 features from a single one.
