# Data Science Portfolio by Dimitrios Effrosynidis

This portfolio is a compilation of notebooks which I created for Data Science related tasks like Tutorials, Exploratory Data Analysis, and Machine Learning.
More notebooks will be added as I learn things and devote time to write about them.

Visit [my website](https://deffro.github.io/), where I include everything listed here and much more.

Below it is a summary of them.

## :fire: Exploratory Data Analysis for the popular Battle Royale game PUBG

This is a very popular [kaggle kernel](https://www.kaggle.com/deffro/eda-is-fun) with more than 800 upvotes and 30.000 views, with which I won the **1st prize** for the best kernel in that Kaggle competition.

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

In [this](https://github.com/Deffro/Data-Science-Portfolio/blob/master/Notebooks/Feature%20Engineering%20with%20Dates.ipynb) tutorial I present the datetime format that Pandas provides to handle datetime features. In the end I create a function that generates 23 features from a single one.
