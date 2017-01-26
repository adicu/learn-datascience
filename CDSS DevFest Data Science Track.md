<a id="top"></a>
# CDSS DevFest Data Science Track

*A short introduction to machine learning in python covering data manipulation, visualization, and basic supervised & unsupervised algorithms.*

Written and developed by [Lucas Schuermann](http://lvs.io), [CDSS](http://cdssatcu.com) and [ADI](adicu.com).

<a href="#top" class="top" id="table-of-contents">Top</a>
## Table of Contents

-	[1.0 Section](#section)
	-	[1.1 Subsection](#subsection)
-	[2.0 Another Section](#another-section)
	-	[2.1 Another Subsection](#another-subsection)
-   [Additional Resources](#additionalresources)


------------------------------
<a href="#top" class="top" id="environment_setup">Top</a>
## 0.0 Environment Setup

While some computers come with Python already installed, such as those running macOS, we will need a number of other Python tools (packages) in order to effectively manipulate, analyze, and visualize data. This guide will walk you through the easiest ways to set up your own data science environment, from installing python to running your first Jupyter notebook.

<a id="using_conda"></a>
### 0.A Using Conda

We highly recommend that you download and use
[Conda](https://www.continuum.io/downloads), an open-source package
manager for scientific computing. Why? Well, scientific computing
packages typically link to some heavy-duty C and Fortran libraries.
Conda makes it extremely easy to install these cross-language
dependencies easily on all major platforms. If you don't use Conda,
you might have to spend time messing around with installing an entire
Fortran toolchain, which is definitely not fun (especially on Windows). This might take a long time to download so make sure you have a strong wifi signal!

The easiest way to install Conda is with Anaconda (you should pick python 2), which is essentially Conda bundled with two dozen commonly
installed dependencies. Anaconda comes with
[Spyder](https://en.wikipedia.org/wiki/Spyder_%28software%29) a nice IDE
for Python data science.  We will also use [Jupyter
notebooks](http://jupyter.org/), which let you code in the browser (this
curriculum was originally developed on a Jupyter notebook, and we do
make use of a couple Jupyter-specific commands).

Be warned, though: Anaconda installs a _lot_ of packages.  If you're
strapped for hard drive space, consider installing
[Miniconda](http://conda.pydata.org/miniconda.html), the minimal
distribution with Conda.
After installing Conda, open up a command prompt (`cmd.exe` for Windows
and `Terminal` for Mac) and type:

```bash
$ conda update conda
$ conda install bokeh matplotlib notebook numpy pandas requests scikit-learn seaborn
```

For Conda power users, you also use the following `environment.yml` file:

```
name: devfest
dependencies:
- beautifulsoup4=4.4
- matplotlib=1.5.1
- notebook=4.1.0
- numpy=1.10.2
- pandas=0.17.1
- python=3.5
- requests=2.9.1
- seaborn=0.7.0
```

<a id="without_conda"></a>
### 0.B Without Conda

We highly suggest using conda for easy package management. You should only be using python without conda if you do not have root access to your machine.

If you're not using Conda, then we recommend you install Python 2.7 from
the [Python website](https://www.python.org/). Certain computers may
come with a version of Python installed: if you see something like:

```bash
$ python
Python X.X.X (default, Dec  7 2015, 11:16:01)
[GCC 4.8.4] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

when you type `python` onto a terminal, then you're good to go (don't
worry too much if you have Python 2.7.A or Python 2.7.B -- our code should be compatible with all versions).

Python should automatically come with `pip`, the Python package
manager. We will install a set of packages commonly used for data science and visualization. Open up a terminal and type:

```bash
$ pip install ipython
$ pip install jupyter
$ pip install matplotlib
$ pip install numpy
$ pip install pandas
$ pip install scikit-learn
$ pip install scipy
$ pip install seaborn
```

On Windows, you might need to visit [Christoph Gohlke's Unofficial
Windows Binaries](http://www.lfd.uci.edu/~gohlke/pythonlibs/) to get
things to install correctly.

## 1. Getting Started

Open a terminal on macOS or linux (or `cmd.exe` on Windows) and run:

```bash
$ jupyter notebook
```

a Jupyter notebook window should pop-up. Just create a new Python 2 notebook and you should be good to go.

If you decide to use something other than a Jupyter notebook, note that we used some Jupyter-specific commands. We wrote this curriculum using [Jupyter notebooks](http://jupyter.org/), so there may be some slight finnegaling required (such as omitting `%matplotlib inline`). 

In order to access datasets bundled with this curriculum, as well as view it in the original notebooks, it is recommended to "clone" it using git. If your computer does not have git, or you are not familiar with command line interfaces, please download it [here](https://github.com/cerrno/intro-to-datascience), else use the following command in your favorite projects folder:

```bash
$ cd [your_favorite_projects_directory]
$ git clone https://github.com/cerrno/intro-to-datascience.git
```

Hint: the download link for a zip folder can be found on the github project page as follows

<img src="files/Github Instructions.png">

___________
<a href="#top" class="top" id="loading_data">Top</a>
## 1.0 Loading and Exploring Data

<a id="another-subsection"></a>
### 2.1 Another Subsection

___________
<a href="#top" class="top" id="basic_visualization">Top</a>
## 2.0 Basic Data Visualization with matplotlib

Data visualization is an important part of any data science project. Whether you use it to give yourself a better understanding of the data that you're about to work with or to make your conclusions more digestible for others, data visualization can add clarity in a way that sheer rows and columns of data cannot.

<a id="2.1"></a>
### 2.1 Setup

```python
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
```
<a id="2.2"></a>
### 2.2 Import Data

The data we'll be working with comes from http://www.makeovermonday.co.uk/, a website that features weekly data visualization challenges in which they present a data visualization in a news article as well as the underlying data, and they give users a week to transform the visualization into something more interesting. In particular, we'll be working with prices of common holiday dinner foods found here: https://onedrive.live.com/view.aspx?resid=43EBDBC5D5265516!10892&ithint=file%2cxlsx&app=Excel&authkey=!AMjmJ1zXKzkIbKg

```python
# Load the data from wherever you stored it, noting that we'll be using Product Family as our index column
food_data = pd.read_excel('datasets/Xmas Food Prices.xlsx', index_col = 'Product Family')
```

```python
# Have a look at the first 5 rows
food_data.head()
```

<a id="2.3"></a>
### 2.3 Make a Scatter Plot

We'll start off by making a simple scatter plot. Let's say you're a whisky fan and you're concerned about the change in price over time. Let's create a scatter plot that shows years on the x-axis and whisky price on the y-axis to see if your concerns are valid.

```python
# Create a list of years using the columns from your data
years = list(food_data)

# Select the whisky prices 
whisky = food_data.ix['Blended Whisky']
```

```python
# Now make the scatter plot
plt.scatter(years, whisky)
```

You can now see that there is unfortunately a notable increase in whisky prices since 2006. That said, say you're also a big fan of parsnips. Let's have a look at a single plot combining both whisky and parsnip prices to see how they stack up.

```python
# Create another set of data for parsnips
parsnips = food_data.ix['Parsnips']
```

```python
# There are lots of parameters you can use to spice up your visualizations.
# We'll use a simple one, noting the color that we want our different marks to be.
plt.scatter(years, whisky, color = 'red')
plt.scatter(years, parsnips, color = 'blue')
plt.show()
```

Good news! The price of parsnips is holding fairly steady. This is probably better for your long-term health than the whisky was anyway.

Now let's say you want to share this chart with your fellow whisky and parsnip-loving friends. There's some work to be done here if anyone else is going to understand this graphic on its own.

```python
# Keep the data points as before, adding a label
plt.scatter(years, whisky, color = 'red', label = 'whisky')
plt.scatter(years, parsnips, color = 'blue', label = 'parsnips')

# Start off by giving it a title
plt.title("Whisky and Parsnip Prices")

# Give the x and y-axes labels
plt.xlabel('Year')
plt.ylabel('Price in GBP')

# Add a legend so it's clear which point is which
plt.legend(loc='upper left')

plt.show()
```

Much better! And that's only the beginning of what you can do. You could add more data, call out specific points, change the range of the axes, and plenty more.

<a id="2.4"></a>
### 2.4 Make a Bar Plot

Let's look at the data another way. Say that instead of looking at the price of a few items over time, you're interested in all the items for 2016. A scatter plot no longer makes much sense because the x-axis is a series of unordered food items instead of ordered years. A bar plot makes more sense in this situation.

```python
# Not a necessary step, but a helpful one. This converts our column labels (like 2016) from integers to strings
food_data.columns = food_data.columns.astype(str)

# Let's have a look at just the data from 2016
year_data = food_data['2016']
year_data
```

This is a good start, albeit including some pretty strange food choices. We'll have a quick review of your pandas lesson to remove the final row of 'Totals' so that doesn't throw off the scale, then move forward with creating our graph.

```python
# Remove the last row (row number -1 in python terms)
year_data = year_data[:-1]
```

```python
year_data.plot(kind='bar')
```

This isn't bad, but is also a little bit hard to read. Let's instead create a horizontal bar chart using "barh" instead of "bar"

```python
year_data.plot(kind='barh')
```

Just like we saw with the scatter plots, you have tons of options for adding style to your bar plots. You can add a title, labels, colors, and all sorts of other details in a similar way as before. You can search through the matplotlib [website](http://matplotlib.org/1.2.1/index.html) for more information.

<a id="2.5"></a>
### 2.5 Make a Histogram

Now let's suppose you want to see how many items fall into different sets of price ranges. You can do this using a histogram, which is just as simple as the graphs we've seen above. Let's use the 2016 data again:

```python
# The first parameter is the data; the second represents the ranges for your bins.
# Bin ranges can be equal as below or have any degree of separation you want.
plt.hist(year_data, bins=[0, 5, 10, 15, 20, 25, 30])
```

You can see that the vast majority of items fall within the cheapest bin, with a couple of outliers on the very expensive end.

___________
<a href="#top" class="top" id="training_models">Top</a>
## 3.0 Training Models on Datasets

___________
<a href="#top" class="top" id="supervised_learning">Top</a>
## 4.0 Supervised Learning Problem

Supervised learning is a machine learning problem where you can train a model to use features to predict a target variable based on prior examples.  This contrasts with unsupervised learning (eg. clustering), in which the data contains many features but no apparent target variable.

To get a taste for supervised learning, we'll build a random forest model to predict secondary school student behavior using each student's attributes.

<a id="4.1"></a>
### 4.1 Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
```

<a id="4.2"></a>
### 4.2 What is supervised learning?

In machine learning, you have a problem that needs to be solved, data related to the problem, and a computer.  In supervised learning, you're feeding data to the computer and training it to output the correct answer for a given set of inputs.

> For example, let's say you love eating at Jin Ramen but hate waiting for a seat.  Let's also suppose that you have a dataset of 500 visits to Jin Ramen that contains information about the wait time for a table, the time of arrival, day of week, number of people in the party, etc.  You could train a supervised learning model on this dataset to predict wait time based on the other variables.  In the future, whenever you want ramen, you can input the values you know (the "features") into this model to predict wait time (the "target"), then decide whether to visit Jin Ramen or not.

Supervised learning usually involves regression and classification tasks:
- Regression: the target variable is continuous (eg. predicting the minutes you'll wait for a table)
- Classification: the target variable is a discrete value (eg. your waiting limit is 10 mins, so you train the model to output "0" if the wait time will be over 10 mins, and "1" if the wait time will be 10 mins or under)

<a id="4.3.0"></a>
### 4.3.0 Introduction to random forest

Random forest is a machine learning technique used for classification and regression tasks.  Random forest is an "ensemble" method because one model is actually composed of many decision trees, each of which produces an output; these outputs are then averaged to produce the random forest model's final prediction.

<a id="4.3.1"></a>
#### 4.3.1 What's a decision tree?
A decision tree is a model that predicts the target value by running the inputs through a tree structure.  Decision trees can perform regression and classification tasks, so they're pretty flexible when it comes to machine learning tasks.

This image is a simple decision tree for predicting whether a Titanic passenger "survived" or "died" (those are the target values in this classification task):
![decision tree](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)

<a id="4.3.2"></a>
### 4.3.2 Lots of decision trees = Forest
A random forest model is composed of many (you decide how many) decision trees.  The idea is that one decision tree may not be accurate, but an entire forest of independent trees will provide higher accuracy because noise is canceled out and signal is strengthened.

When creating each tree, the random forest algorithm takes a sample of the data (a bootstrap sample), and produces the best possible decision tree for that sample.  When building the tree, at every node/split, a random sample of features is used.  All this sample-taking allows the forest of trees to identify data features that are truly important, not arbitrary.

Once the entire forest is created and ready to make predictions, the random forest model runs the input down every tree, and outputs the average or majority decision of its trees.

> Pros:
- High accuracy
- Fast to train
- Can handle many input variables, which relieves you of having to choose a subset of all variables before training the model
- Can work with numeric and categorical variables

> Cons:
- Hard to interpret
- Prone to overfitting on noisy datasets
- Cannot deal with features or target values that do not exist in the training data

<a id="4.3"></a>
### 4.3 Random forest example

We're going to predict student alcohol consumption using a dataset about Portuguese students.  Take a moment to skim over the variables [here](http://archive.ics.uci.edu/ml/datasets/STUDENT+ALCOHOL+CONSUMPTION).

For the training the random forest model, we're going to use the built-in functions of `scikit-learn`.  Documentation for its random forest classifier is [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

```python
# Load in data from the datasets folder
student_port = pd.read_csv('datasets/student-por.csv', sep = ';')
```

```python
# Take a look at the first 5 rows
student_port.head()
```

```python
# Get the dataset's dimensions
student_port.shape
```

Let's try to predict alcohol consumption based on all or most of other variables (a classification problem).

First, we need to perform some data cleaning (highly important for any data task!).  Off the bat, we can see that some of the columns don't have the correct data type.  We want to make sure every column is correctly stored as a numeric or categorical data type.

```python
# What's the current data type of each variable?
student_port.info()
```

```python
# Most variables should be categorical, so let's list out the variables that are numeric and/or nominal
vars_num = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
```

```python
# Let's put all the rf data into a new df, rf_data
rf_data = student_port[vars_num]
```

```python
# But need to convert every categorical value into a number
# Label encoder converts to 0, 1, ... for alphabetically-sorted string values
for col in student_port.columns:
    if col not in vars_num:
        le = preprocessing.LabelEncoder()
        rf_data.loc[:, col] = le.fit_transform(student_port.loc[:, col])
        rf_data.loc[:,col] = rf_data.loc[:, col].astype('category')
```

```python
# If we want to know the string value corresponding to a numerical category in X, go back to student_port
sorted(student_port['address'].unique())
```

```python
# How many students in each category of workday alcohol consumption?
student_port['Dalc'].value_counts()
```

```python
# How many students in each category of weekend alcohol consumption?
student_port['Walc'].value_counts()
```

Five categories of student alcohol consumption may be too many to work with (especially since most students fall on the low end of consumption).  Since it's more interesting to divide students into light drinkers and heavy drinkers, let's create two categories of alcohol usage: 1-2 and 3-5.

```python
# Map out the old level to the new level
# 0 will be "light drinkers", 1 will be "heavy drinkers"
alc_mapping = {1: 0, 2: 0, 3: 1, 4: 1, 5:1}
```

```python
# Create new columns for the mapped data
rf_data['Dalc_mapped'] = rf_data['Dalc'].map(alc_mapping)
rf_data['Walc_mapped'] = rf_data['Walc'].map(alc_mapping)
```

<a id="4.4.0"></a>
### 4.4.0 Modeling time!

Supervised learning involves these general steps:

1. Split the data into training and test datasets.  Usually the split is anywhere from 60%/40% to 80%/20%.
2. Train a model on the training set.
3. Apply the model on the test set and compare the predicted results against the actual values to evaluate the model's performance.
    - As needed, iterate on the model to improve its performance (beware of overfitting!)
    - Look into the model to understand what it's doing, and gain some insight on your dataset.

<a id="4.4.1"></a>
#### 4.4.1 Decide on the data you want, and split into training and test sets

```python
# Specify the columns you want as features and as the target
features = rf_data.columns
features = features.drop(['G1', 'G2', 'Mjob', 'Fjob', 'Dalc', 'Walc', 'Dalc_mapped', 'Walc_mapped'])
target = 'Walc_mapped'
```

```python
# This is the full dataset of predictor variables
X = rf_data[features]

# This is the column of target (to be predicted) variables
y = rf_data[target]

# Split into training (70%) and test (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
```

```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

<a id="4.4.2"></a>
#### 4.4.2 Train the random forest model

```python
# Initialize the random forest model with your desired parameters
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 2017)

rf_model = rf_model.fit(X_train, y_train)
```

<a id="4.4.3"></a>
#### 4.4.3 Evaluate the model

How well does the model perform?

```python
# Model accuracy: the number of correct predictions divided by the number of predictions
rf_model.score(X_test, y_test)
```

```python
# Apply the model on the test data to produce predictions
y_pred = rf_model.predict(X_test)
```

```python
# A confusion matrix helps you evaluate the predictions (rows) against the actual values (columns)
confusion_matrix(y_test, y_pred)
```

```python
# This is a pretty visualization of the confusion matrix
# From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

```python
plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes = sorted(y.unique()))
```

Which features did the model deem most "important" for predicting the target variable?

*Note:* Random forest models are considered "black box" models because they're hard to interpret.  If our model only had one decision tree instead of an entire forest of them, then we could examine the tree and see the exact criteria used to produce a prediction for any input.

```python
# Show feature importances, highest first
pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_}).sort_values(
    by = 'importance', ascending = False)
```

Intuitively, does it make sense that the top few features are most important in determining a student's alcohol level consumption?

<a id="4.5"></a>
### 4.5 Next steps: Different ways to play around with random forests

- Tweak the random forest initialization parameters: change the number of decision trees
- Use a different set of features in X
- Predict a different variable (eg. student's test scores)
    - Use RandomForestRegressor instead of RandomForestClassifier to predict a variable that is numeric and continuous rather than categorical
- Transform a variable in a different way (eg. a different mapping of variables; combine variables with each other)
- Use random forests's built-in out-of-bag error to evaluate model performance
    - Out-of-bag estimate is the error rate of the random forest model on the training data that is not included in the bootstrap sample of each tree.  Oob error has been shown to be a good measure of error for random forest models.
    - When initializing `RandomForestClassifier`, set `oob_score = True`

<a id="4.6"></a>
### 4.6 Further resources

- Another random forest tutorial: [Random Forests in Python](http://blog.yhat.com/posts/random-forests-in-python.html)
- [Would You Survive the Titanic? A Guide to Machine Learning in Python](https://blog.socialcops.com/engineering/machine-learning-python/)
- [Supervised learning with scikit-learn](http://scikit-learn.org/stable/supervised_learning.html)

___________
<a href="#top" class="top" id="unsupervised_learning">Top</a>
## 5.0 Unsupervised Learning Problem

[github]: https://github.com/cerrno/intro-to-datascience.git
[learn]: http://adicu.com/learn
[codecademy]: http://www.codecademy.com
 
