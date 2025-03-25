# Converted from: DecisionTreesVisualization.ipynb

# Markdown Cell 1
# <h1 align="center"> Visualizing Decision Trees with Python (Scikit-learn, Graphviz, Matplotlib) </h1>

# Markdown Cell 2
# ## How to Fit a Decision Tree Model using Scikit-Learn

# Markdown Cell 3
# In order to visualize decision trees, we need first need to fit a decision tree model using scikit-learn. If this section is not clear, I encourage you to read my [Understanding Decision Trees for Classification (Python) tutorial](https://towardsdatascience.com/understanding-decision-trees-for-classification-python-9663d683c952) as I go into a lot of detail on how decision trees work and how to use them.

# Markdown Cell 4
# ### Import Libraries

# Cell 5
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

# Markdown Cell 6
# ### Load the Dataset
# The Iris dataset is one of datasets scikit-learn comes with that do not require the downloading of any file from some external website. The code below loads the iris dataset.

# Cell 7
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

# Markdown Cell 8
# ### Splitting Data into Training and Test Sets

# Markdown Cell 9
# ![images](../images/trainTestSplit.png)
# The colors in the image indicate which variable (X_train, X_test, Y_train, Y_test) the data from the dataframe df went to for a particular train test split. Image by [Michael Galarnyk](https://twitter.com/GalarnykMichael).

# Cell 10
X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)

# Markdown Cell 11
# ### Scikit-learn 4-Step Modeling Pattern
# 
# <b>Step 1:</b> Import the model you want to use
# 
# In sklearn, all machine learning models are implemented as Python classes

# Cell 12
# This was already imported earlier in the notebook so commenting out
#from sklearn.tree import DecisionTreeClassifier

# Markdown Cell 13
# <b>Step 2:</b> Make an instance of the Model

# Cell 14
clf = DecisionTreeClassifier(max_depth = 2, 
                             random_state = 0)

# Markdown Cell 15
# <b>Step 3:</b> Training the model on the data, storing the information learned from the data

# Markdown Cell 16
# Model is learning the relationship between x (features: sepal width, sepal height etc) and y (labels-which species of iris)

# Cell 17
clf.fit(X_train, Y_train)

# Markdown Cell 18
# <b>Step 4:</b> Predict the labels of new data (new flowers)
# 
# Uses the information the model learned during the model training process

# Markdown Cell 19
# Uses the information the model learned during the model training process

# Cell 20
# Returns a NumPy Array
# Predict for One Observation (image)
clf.predict(X_test.iloc[0].values.reshape(1, -1))

# Markdown Cell 21
# Predict for Multiple Observations (images) at Once

# Cell 22
clf.predict(X_test[0:10])

# Markdown Cell 23
# ### Measuring Model Performance

# Markdown Cell 24
# <b>This part is not in the blog post, but I figured I would include it.</b> While there are other ways of measuring model performance (precision, recall, F1 Score, [ROC Curve](https://towardsdatascience.com/receiver-operating-characteristic-curves-demystified-in-python-bd531a4364d0), etc), we are going to keep this simple and use accuracy as our metric. 
# To do this are going to see how the model performs on new data (test set)
# 
# Accuracy is defined as:
# (fraction of correct predictions): correct predictions / total number of data points

# Cell 25
score = clf.score(X_test, Y_test)
print(score)

# Markdown Cell 26
# ## How to Visualize Decision Trees using Matplotlib

# Markdown Cell 27
# As of scikit-learn version 21.0 (roughly May 2019), Decision Trees can now be plotted with matplotlib using scikit-learn's `tree.plot_tree` without relying on the dot library which is a relatively hard-to-install dependency. 

# Cell 28
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)

tree.plot_tree(clf);
fig.savefig('../images/plottreedefault.png')

# Cell 29
# Putting the feature names and class names into variables
fn = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn = ['setosa', 'versicolor', 'virginica']

# Cell 30
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)

tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('../images/plottreefncn.png')

# Markdown Cell 31
# ## How to Visualize Decision Trees using Graphviz

# Markdown Cell 32
# ![images](../images/graphvizCTblog.png)
# The image above is a Decision Tree I produced through Graphviz. Graphviz is open source graph visualization software. Graph visualization is a way of representing structural information as diagrams of abstract graphs and networks. In data science, one use of Graphviz is to visualize decision trees. I should note that the reason why I am going over Graphviz after covering Matplotlib is that getting this to work can be difficult. The first part of this process involves creating a dot file. A dot file is a Graphviz representation of a decision tree. The problem is that using Graphviz to convert the dot file into an image file (png, jpg, etc) can be difficult. There are a couple ways to do this including: installing python-graphviz though Anaconda, installing Graphviz through Homebrew (Mac only), installing Graphviz through executables (Windows only), and using an online converter on the content of your dot file to convert it into an image.
# ![images](../images/dot2imagefile.png)

# Markdown Cell 33
# ### Export your model to a dot file
# The code below code will work on any operating system as python generates the dot file and exports it as a file named tree.dot.

# Cell 34
tree.export_graphviz(clf,
                     out_file="tree.dot",
                     feature_names = fn, 
                     class_names=cn,
                     filled = True)

# Cell 35
# Ignore this cell as I am just rotating the decision tree output. 
"""
tree.export_graphviz(clf,
                     out_file="treeRotated.dot",
                     feature_names = fn, 
                     class_names=cn,
                     rotate = True,
                     filled = True)
"""

# Markdown Cell 36
# ### Installing and Using Graphviz
# Converting the dot file into an image file (png, jpg, etc) typically requires installation of Graphviz which depends on your operating system and a host of other things. I highly recommend that if you get an error in the code below to see the blog and see how to install it on your operating system.

# Cell 37
!dot -Tpng -Gdpi=300 tree.dot -o tree.png

# Cell 38
#!dot -Tpng -Gdpi=300 treeRotated.dot -o treeRotated.png

# Markdown Cell 39
# ## How to Visualize Individual Decision Trees from Bagged Trees or Random Forests

# Markdown Cell 40
# ![images](../images/BaggedTrees.png)
# 
# A weakness of decision trees is that they don't tend to have the best predictive accuracy. This is partially because of high variance, meaning that different splits in the training data can lead to very different trees.
# 
# The image above could be a diagram for Bagged Trees or Random Forests models which are ensemble methods. This means using multiple learning algorithms to obtain a better predictive performance than could be obtained from any of the constituent learning algorithms alone (many trees protect each other from their individual errors). How exactly Bagged Trees and Random Forests models work is a subject for another blog, but what is important to note is that for each both models we grow N trees where N is the number of decision trees a user specifies. Consequently after you fit a model, it would be nice to look at the individual decision trees that make up your model.

# Markdown Cell 41
# ### Load the Dataset
# The Breast Cancer Wisconsin (Diagnostic) Dataset is one of datasets scikit-learn comes with that do not require the downloading of any file from some external website. The code below loads the dataset.

# Cell 42
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

# Markdown Cell 43
# ### Arrange Data into Features Matrix and Target Vector

# Cell 44
X = df.loc[:, df.columns != 'target']

# Cell 45
y = df.loc[:, 'target'].values

# Markdown Cell 46
# ### Split the data into training and testing sets

# Cell 47
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)

# Markdown Cell 48
# ### Random Forests in `scikit-learn` (with N = 100)

# Markdown Cell 49
# <b>Step 1:</b> Import the model you want to use
# 
# In sklearn, all machine learning models are implemented as Python classes

# Cell 50
# This was already imported earlier in the notebook so commenting out
# from sklearn.ensemble import RandomForestClassifier

# Markdown Cell 51
# <b>Step 2:</b> Make an instance of the Model

# Cell 52
rf = RandomForestClassifier(n_estimators=100,
                            random_state=0)

# Markdown Cell 53
# <b>Step 3:</b> Training the model on the data, storing the information learned from the data. Model is learning the relationship between features and labels

# Cell 54
rf.fit(X_train, Y_train)

# Markdown Cell 55
# <b>Step 4:</b> Predict the labels of new data
# 
# Uses the information the model learned during the model training process

# Cell 56
# Not doing this step in the tutorial
# class predictions (not predicted probabilities)
# predictions = rf.predict(X_test)

# Markdown Cell 57
# ### Measuring Model Performance

# Markdown Cell 58
# <b>This part is not in the blog post, but I figured I would include it.</b> While there are other ways of measuring model performance (precision, recall, F1 Score, [ROC Curve](https://towardsdatascience.com/receiver-operating-characteristic-curves-demystified-in-python-bd531a4364d0), etc), we are going to keep this simple and use accuracy as our metric. 
# To do this are going to see how the model performs on new data (test set)
# 
# Accuracy is defined as:
# (fraction of correct predictions): correct predictions / total number of data points

# Cell 59
# In this dataset we have 357 benign and 212 malignant
score = rf.score(X_test, Y_test)
print(score)

# Cell 60
data.target_names

# Markdown Cell 61
# ### Visualizing your Estimators

# Markdown Cell 62
# You can now view all the individual trees from the fitted model.

# Cell 63
rf.estimators_

# Cell 64
# We have 100 estimators
print(len(rf.estimators_))

# Cell 65
rf.estimators_[0]

# Markdown Cell 66
# You can now visualize individual trees.

# Cell 67
fn=data.feature_names
cn=data.target_names
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')

# Markdown Cell 68
# You can try to use Matplotlib subplots to visualize the first 5 decision trees. I personally don't prefer this method as I find it doesn't save the output very well (particularly as visualizing decision trees through Matplotlib is a relatively new feature) and it is even harder to read.

# Cell 69
# This may not the best way to view each estimator as it is small 

fn=data.feature_names
cn=data.target_names
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=1000)

for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);
    
    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)


fig.savefig('rf_5trees.png')

# Markdown Cell 70
# ### Create Images for each of the Decision Trees (estimators)

# Cell 71
# This code is just making the feature name text shorter
fn = []
for feature in data.feature_names: 
    parts = []
    for part in feature.split(): 
        parts.append(part.title())
    ''.join(parts)
    fn.append(''.join(parts))

cn=data.target_names

for index in range(0, len(rf.estimators_)):
    #plt.figure(figsize = (4,4), dpi = 300)
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 900)
    tree.plot_tree(rf.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True)
    
    importances = pd.DataFrame({'feature':fn,'importance':np.round(rf.estimators_[index].feature_importances_,2)})
    importances = importances.sort_values('importance',ascending=False)
    
    axes.set_title('Estimator: ' + str(index) + 
              '\n\'Most Important\' Feature:' + importances.iloc[0]['feature'] +
              '(' + str(importances.iloc[0]['importance']) +  ')', fontsize = 10.5)

    
    fig.savefig('../imagesanimation/' + 'initial' + str(index).zfill(4) + '.png')
    plt.close('all')

# Markdown Cell 72
# I highly recommend that you DONT TRY AND DO THIS ON YOUR COMPUTER. It might slow down your computer a lot. This also assumes you have ffmpeg. For my mac, I did `brew install ffmpeg`. Here is an okay reference on installing it on windows (https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg)

# Cell 73
!ffmpeg -framerate 1 -i '../imagesanimation/initial%04d.png' -c:v libx264 -r 30 -pix_fmt yuv420p '../imagesanimation/initial_002.mp4'

