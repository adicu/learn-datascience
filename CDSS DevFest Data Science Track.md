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

<a id="0.1"></a>
### 0.1 Getting Started

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
## 5.0 Unsupervised Learning: k-Means Clustering

We hope you enjoy the tutorial! Before we start diving into the material, let's make sure that you have your environment up and running. Simply run the code below -- if things break, you can install the dependencies using pip or conda.

<a id="5.1"></a>
### 5.1 Setup

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
```

<a id="5.2"></a>
### 5.2 What's Unsupervised Learning?

The basic notion behind machine learning is that you're given a dataset with an interesting backstory, and it's up to you to figure out what that story is.  Maybe you want to predict the next big thing that will break the stock market, or understand the relationship between students' stress levels and pounds of chocolate consumption.  In both cases, you're looking at the interactions of several different things and uncovering the hidden patterns that allow you to draw insightful conclusions from this data.

We can break down such problems into two categories: supervised and unsupervised.
- Supervised learning is when your explanatory variables X come with an associated reponse variable Y.  You get a sneak peak at the true "labels": for example, for all the participants in a clinical trial, you're told whether their treatments were successful or not.
- In *unsupervised learning*, sorry -- no cheating.  You get a bunch of X's without the Y's.  There's some ground truth we don't have access to.  So we have to do our best to extract some meaning out of the data's underlying structure and check to make sure that our methods are robust. 

One example of an unsupervised learning algorithm is clustering, which we'll practice today!

<a id="5.3"></a>
### 5.3 Clustering

Clustering is what it sounds like: grouping “similar” data points into *clusters* or *subgroups*, while keeping each group as distinct as possible.  The data points belonging to different clusters should be different from each other, too.  Often, we'll come across datasets that exhibit this kind of grouped structure.  **k-Means** is one of many ways to perform clustering on your data.

But wait -- these are vague concepts.  What does it mean for two data points to be "similar?"  And are we actually moving the points around physically when we group them together? 

These are all good questions, so let’s walk through some vocab before we walk through the steps of the k-means clustering algorithm:

> #### (a) Similarity
Intuitively, it makes sense that similar things should be close to each other, while different things should be far apart.  To formalize the notion of **similarity**, we choose a **distance metric** (see below) that quantifies how "close" two points are to each other.  The most commonly used distance metric is Euclidean distance (think: distance formula from middle school), and that's what we'll use in our example today.  We'll introduce other distance metrics towards the end of the notebook. 

> #### (b) Cluster centroid
The **cluster centroid** is the most representative feature of an entire cluster.  We say "feature" instead of "point" because the centroid may not necessarily be an existing data point of the cluster.  To find a cluster's centroid, average the values of all the points belonging to that cluster.  Thus, the cluster centroid gives nice summary information about all the points in its cluster.  Think of it as the cluster's (democratic) president.

<a id="5.4"></a>
### 5.4 The k-means Algorithm

The k-means algorithm has a simple objective: given a set of data points, it tries to separate them into *k* distinct clusters. It uses the same principle that we mentioned earlier: keep the data points within each cluster as similar as possible. You have to provide the value of **k** to the algorithm, so you should have a general idea of how many clusters you expect to see in the data. 

Let’s start by tossing all of our data points onto the screen to see what the data actually looks like. This kind of exploratory data visualization can provide a rough guide as to how to start clustering our data. Remember that clustering is an *unsupervised learning method*, so we’re never going to have a perfect answer for our final clusters. But let’s do our best to get results that are **reasonable** and **replicable**.
- Replicable: someone else can arrive at our results from a different starting point
- Reasonable: our results show some correlation with what we expect to encounter in real life

Let's take a look at a toy example:

![alt-text](http://pubs.rsc.org/services/images/RSCpubs.ePlatform.Service.FreeContent.ImageService.svc/ImageService/Articleimage/2012/AN/c2an16122b/c2an16122b-f3.gif "k-means clustering algorithm")

> ##### (a) 
Our data seems to have some sort of underlying structure. Let’s use this information to initialize our k-means algorithm with k = 3 clusters. For now we assume that we know how many clusters we want, but we’ll go into more detail later about relaxing this assumption and how to choose “the best possible k”. 

> ##### (b) 
We want 3 clusters, so first we randomly “throw down” three random cluster centroids. Every iteration of k-means will eventually "correct" them towards the right clusters. Since we are heading to a correct answer anyway, we don't care about where we start. 

> These centroids are our “representative points” -- they contain all the information that we need about other points in the same cluster. It makes sense to think about centroids as being the physical center of each cluster. So let’s pretend that our randomly initialized cluster centers are the actual centroids, and group our points accordingly. Here we use our distance metric of choice -- Euclidean distance. For every data point, we compute its distance to each centroid, and assign the data point whichever centroid is closest (smallest distance).

> ##### (c)
Now we have something that’s starting to resemble three distinct clusters! But we need to update the centroids that we started with -- we’ve just added in a bunch of new data points to each cluster, so we need our “representative point,” the centroid, to reflect that. 

> ##### (d)-(e)-(f)-(g)
Let's average all the values within each cluster and call that our new centroid. These new centroids are further "within" the data than the older centroids. 

> Notice that we’re not quite done yet -- we have some straggling points which don’t seem to belong to any cluster. Let’s run another iteration of k-means and see if that separates out the clusters better. This means that we’re computing the distances from each data point to every centroid, and re-assign those that are closer to centroids of another cluster.

> ##### (h)
We keep computing the centroids for every iteration using the steps (c) and (d). After a few iterations, maybe you notice that the clusters don’t change after a certain point. This actually turns out to be a good criterion for stopping the cluster iterations!

> There’s no need to keep running the algorithm if our answer doesn't change after a certain point in time. That's just wasting time and computational resources. We can formalize this idea of a “stopping criterion.” We define a small value, call it “epsilon”, and terminate the algorithm when the change in cluster centroids is less than epsilon. This way, epsilon serves as a measure of how much error we can tolerate.   

<a id="5.5"></a>
### 5.5 Image Segmentation Example

Let's move on to a real-life example. You can access images in the `datasets/kmeans/imgs` folder. 

- We know that images often have a few dominant colors -- for example, the bulk of an image is often made up of the foreground color and background color.

- In this example, we'll write some code that uses `scikit-learn`'s k-means clustering implementation to find what these dominant colors may be. `scikit-learn`, or `sklearn` for short, is a package of built-in machine learning algorithms all coded up and ready to use. 

- Once we know what the most important colors are in an image, we can compress (or "quantize") the image by re-expressing the image using only the set of k colors that we get from the algorithm. Let's try it!

```python
# let's list what images we have to work with
imgs = os.listdir('datasets/kmeans/imgs/')
print(imgs)
```

Let's use an image of Leo's beautiful, brooding face for our code example.

```python
img_path = os.path.join('datasets/kmeans/imgs/', imgs[0])
print('Using image 0: path {}'.format(img_path))

img = mpimg.imread(img_path)

# normalize the image values
img = img * 1.0 / img.max()

imgplot = plt.imshow(img)
```

An image is represented here as a three-dimensional array of floating-point numbers, which can take values from 0 to 1. If we look at ``img.shape``, we'll find that the first two dimensions are x and y, and then the last dimension is the color channel. There are three color channels (one each for red, green, and blue). A set of three channel values at a single (x, y)-coordinate is referred to as a "pixel".

We're going to use a small random sample of 10% of the image to find our clusters.

```python
print('Image shape: {}'.format(img.shape))
width, height, num_channels = img.shape
num_pixels = width * height
num_sample_pixels = num_pixels / 10

print('Sampling {} out of {} pixels'.format(num_sample_pixels, num_pixels))
```

Next we need to reshape the image data into a single long array of pixels (instead of a two-dimensional array of pixels) in order to take our sample.

```python
img_reshaped = np.reshape(img, (num_pixels, num_channels))
img_sample = shuffle(img_reshaped, random_state=0)
```

Now that we have some data, let's construct our k-means object and feed it some data. It will find the best k clusters, as determined by a distance function.

```python
# We're going to try to find the 20 colors which best represent the colors in the picture.
K = 20

t0 = time()
kmeans = KMeans(n_clusters=K, random_state=0)

# actually running kmeans is super simple!
kmeans.fit(img_sample)
print("K-means clustering complete. Elapsed time: {} seconds".format(time() - t0))
```

The center of each cluster represents a color that is significant in the image. We can grab the values of these colors from `kmeans.cluster_centers_`. We can also call `kmeans.predict()` to match each pixel in the image to the closest color, which will let us know the size of each cluster (and also serve as a way to quantize the image)

```python
# There are K cluster centers, each of which is a RGB color
kmeans.cluster_centers_

t0 = time()
labels = kmeans.predict(img_reshaped)
print("k-means labeling complete. Elapsed time: {} seconds".format(time() - t0))
```

```python
# construct a histogram of the points in each cluster
n, bins, patches = plt.hist(labels, bins=range(K+1))

# a bit of magic to color the bins the right color
for p, color in zip(patches, kmeans.cluster_centers_):
    plt.setp(p, 'facecolor', color)
```

As you can tell from the above histogram, the most dominant color in the scene is the background color, followed by a large drop down to the foreground colors. This isn't that surprising, since visually we can see that the space is mostly filled with the background color -- that's why it's called the "background"!

Now, let's redraw the scene using only the cluster centers! This can be used for image compression, since we only need to store the index into the list of cluster centers and the colors corresponding to each center, rather than the colors corresponding to each pixel in the original image.

```python
quantized_img = np.zeros(img.shape)
for i in range(width):
    for j in range(height):
        # We need to do some math here to get the correct
        # index position in the labels array
        index = i * height + j
        quantized_img[i][j] = kmeans.cluster_centers_[labels[index]]

quantized_imgplot = plt.imshow(quantized_img)
```

Note that the image looks similar, but the gradients are no longer as smooth, and there are a few image artifacts scattered throughout. This is because we're only using the k most representative colors, which excludes the steps along the gradient.

Try running the code again with a different image, or with a different value of k!

<a id="5.6.0"></a>
### 5.6.0 Limitations, Extensions and the basis of some assumptions used above

#### I. Choosing the Right K
In the first toy example, we started with k = 3 centroids. If you're wondering how we arrived at this magic number and why, read on. 

> ##### (a) Known number of centroids (Relatively Easy)
Sometimes, you may be in a situation where the number of clusters is provided to you beforehand. For example, you may be asked to categorize a vast range of different bodily actions to the three main subdivisions of the brain (cerebrum, cerebellum, and medulla). Here you know to look for three main clusters where each cluster will represent a specific part of the brain, so you will use three centroids. 
    
> ##### (b) Unknown number of centroids (Hard)
However, you often do not know how many centroids to pick up from your data. Two extreme situations can happen.
* You could end up making every point its own representative (a perfect centroid) at the risk of losing grouping tendencies. This is called the overfitting problem. While each point perfectly represents itself, it gives you no summary information, no insight, about the data as a whole.
* You could end up choosing only one centroid from all the data (a perfect grouping). Since there is no way to generalize an enormous volume of data to one point, this method also fails to produce relevant distinguishing features from the data. This is like saying that all people in the world drink water, so we can cluster them all by this one  feature. In Machine Learning terminology, this is called the underfitting problem.

> ##### (c) How to find how many centroids should be in a cluster? 
Unfortunately, there’s no easy way to determine the optimal value of k. It’s a hard problem: we must balance out the number of clusters that makes the most sense for our data, yet make sure that we don’t overfit our model to the exact dataset at hand. There are a few ways that we can address this:

> The most intuitive explanation is the idea of **stability**. If the clusters we obtain represent a true, underlying pattern in our data, then the clusters shouldn’t change much on separate but similar samples. So if we randomly subsample or split our data into smaller parts and run the clustering algorithm again, the cluster memberships shouldn’t drastically change. If they did, then our clusters might be too finely-tuned to the random noise in our data. Therefore, we can compute “stability scores” for a fixed value of k and observe which value of k produces the most stable clusters. This idea of *perturbation* is also really important for machine learning in general.

> We can also use **penalization approaches**, where we use different criteria such as AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion) to keep the value of k under control. 

> ##### (d) What if we can’t cluster the data?
* In the tutorial above, we try to classify each point into one of K clusters. But sometimes, maybe you are clustering based on a feature that is not so exclusive. For example, people usually enjoy more than one genre of music, or food. It would be pretty difficult to form a clustering system such that a person can be a fan of ice-cream or a fan of tiramisu but not both. Hence when we need to "share" members of clusters, we can use **probabilistic clustering** or **fuzzy clustering**.

<a id="5.6.1"></a>
#### What features should I use? 

If you are still getting questionable results, then the features on which you are trying to cluster may not be good indicators. Consider, for example, a clustering algorithm which clusters people into economic categories based on how many Buzzfeed quizzes they take per week. While you may get a clustering, you know that Buzzfeed quizzes are not empirical indicators of earning potential (personal opinions may differ). The motto is: **garbage metrics will give you garbage clusters**. 

<a id="5.6.2"></a>
#### 5.6.2 Distance Metric

We used the Euclidean distance to find out which points are most closely related to each other. Depending on the distance metric you're using, you can get a different set of clusters. 

The choice of the distance metric depends on the characteristics of your data. For example, distances between "how alike are these faces?" cannot be properly determined by an Euclidean distance metric. DNA and biological data often use non-Euclidean distance metrics. 

[A list of commonly used distance metrics](http://www.mathworks.com/help/stats/kmeans.html?refresh=true)

<a id="5.6.3"></a>
#### 5.6.3 I have too many features. How do I find the most relevant ones to cluster with?

When you encounter too many features to cluster on (and can't choose which one should be the basis of clustering), you can use a machine learning hack called [Principal Components Analysis](http://setosa.io/ev/principal-component-analysis/). While we don't cover PCA here, the takeaway is that PCA can reduce the feature space you work with, by ranking the most relevant eigenvectors of the data in decreasing order of relevance. 

Other reasons to use PCA include:
* You want to plot multi-dimensional data on a 2D graph (PCA equivalent: render only the first two eigenvectors)
* You want to minimize computational cost (PCA gives you fewer features to process. More features = more time to compute distances = less time to celebrate results.)

<a id="5.7"></a>
### 5.7 Other Clustering Algorithms

K-means is one of several unsupervised clustering algorithms, and each method has its strengths and weaknesses. K-means, for example, scales well for the number of data points but not for the number of clusters. It also optimizes for the given number of k, which means that it gives wonky results for values of k that don’t make sense in the context of the data.

We’ll briefly mention two other clustering methods that bypass these challenges. 

> #### (a) Hierarchical Clustering
For hierarchical clustering, there’s no need to worry about the best number of clusters. The algorithm cranks through each data point, grouping the most similar ones together until it ends up with a bunch of small clusters. Then, it clusters these clusters together until you’re left with one giant super-cluster. 

> #### (b) Expectation-Maximization (EM)
EM is *the generalized* clustering algorithm. It views the data as a mixture of different distributions, and tries to find what those distributions are accordingly. It’s a **probabilistic clustering method**, though, in that it doesn’t assign each data point to a fixed cluster. Instead, it determines the probabilities with which each point belongs to each cluster. For example, it may find that a point belongs to cluster 1 with probability 0.95, and cluster 2 with probability 0.05. It iteratively estimates the cluster assignment probabilities and computes the distributional parameters for each cluster.

<a id="5.8"></a>
### 5.8 Further reading

* [Heavy-duty unsupervised learning resource](http://www-users.cs.umn.edu/~kumar/dmbook/ch8.pdf)
* [Bags of Words to Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors) (super fun Kaggle tutorial that you can do!)
* [An Introduction to Statistical Learning in R - James, Witten, Hastie, Tibshirani](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf) (awesome textbook)
* [Elements of Statistical Learning (ESL) - Friedman, Hastie, Tibshirani](http://statweb.stanford.edu/~tibs/ElemStatLearn/)
* [Pattern Recognition and Machine Learning - Bishop](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)

[github]: https://github.com/cerrno/intro-to-datascience.git
[adi_learn]: http://adicu.com/learn
 
