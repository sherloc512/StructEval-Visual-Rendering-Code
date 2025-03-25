# Converted from: ROC_curve.ipynb

# Markdown Cell 1
# # **Machine Learning in Python: Making Receiver Operating Characteristic (ROC) curve**
# 
# Chanin Nantasenamat
# 
# <i>Data Professor YouTube channel, http://youtube.com/dataprofessor </i>
# 
# In this Jupyter notebook, we will be making a Receiver Operating Characteristic (ROC) curve using the Iris data set as an example.

# Markdown Cell 2
# ## **What is ROC curve?**
# 
# The **ROC curve** summarizes the prediction performance of a classification model at all classification thresholds. Particularly, the ROC curve plots the **False Positive Rate (FPR)** on the *X-axis* and the **True Positive Rate (TPR)** on the *Y-axis*.
# 
# $\text{TPR (Sensitivity)} = \frac{TP}{TP + FN}$
# 
# $\text{FPR (1 - Specificity)} = \frac{FP}{TN + FP}$

# Markdown Cell 3
# ## **Generate synthetic dataset**

# Cell 4
from sklearn.datasets import make_classification
import numpy as np

# Cell 5
X, Y = make_classification(n_samples=2000, n_classes=2, n_features=10, random_state=0)

# Markdown Cell 6
# ## **Add noisy features to make the problem more difficult** $^1$

# Cell 7
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Markdown Cell 8
# ## **Data splitting**

# Cell 9
from sklearn.model_selection import train_test_split

# Cell 10
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=0)

# Markdown Cell 11
# ## **Build classification model**

# Cell 12
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Markdown Cell 13
# ### Random Forest

# Cell 14
rf = RandomForestClassifier(max_features=5, n_estimators=500)
rf.fit(X_train, Y_train)

# Markdown Cell 15
# ### Naive Bayes

# Cell 16
nb = GaussianNB()
nb.fit(X_train, Y_train)

# Markdown Cell 17
# ## **Prediction probabilities** $^2$

# Cell 18
r_probs = [0 for _ in range(len(Y_test))]
rf_probs = rf.predict_proba(X_test)
nb_probs = nb.predict_proba(X_test)

# Markdown Cell 19
# Probabilities for the positive outcome is kept.

# Cell 20
rf_probs = rf_probs[:, 1]
nb_probs = nb_probs[:, 1]

# Markdown Cell 21
# ## **Computing AUROC and ROC curve values**

# Cell 22
from sklearn.metrics import roc_curve, roc_auc_score

# Markdown Cell 23
# ### **Calculate AUROC**
# **ROC** is the receiver operating characteristic
# **AUROC** is the area under the ROC curve

# Cell 24
r_auc = roc_auc_score(Y_test, r_probs)
rf_auc = roc_auc_score(Y_test, rf_probs)
nb_auc = roc_auc_score(Y_test, nb_probs)

# Markdown Cell 25
# ### **Print AUROC scores**

# Cell 26
print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Random Forest: AUROC = %.3f' % (rf_auc))
print('Naive Bayes: AUROC = %.3f' % (nb_auc))

# Markdown Cell 27
# ### **Calculate ROC curve**

# Cell 28
r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)

# Markdown Cell 29
# ## **Plot the ROC curve**

# Cell 30
import matplotlib.pyplot as plt

# Cell 31

plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()

# Markdown Cell 32
# ## **Reference**
# 1. https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# 2. https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

# Markdown Cell 33
# ---

