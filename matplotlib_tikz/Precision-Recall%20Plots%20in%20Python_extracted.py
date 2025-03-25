#!/usr/bin/env python
# Extracted from Precision-Recall%20Plots%20in%20Python.md

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Create directory for saved plots
os.makedirs('plots', exist_ok=True)


# Code Block 1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Generate sample data
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.2, 0.9, 0.5, 0.7, 0.3])

# Calculate precision and recall values
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_1.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 2
from sklearn.metrics import precision_score, recall_score

# Calculate precision and recall
precision = precision_score(y_true, y_scores > 0.5)
recall = recall_score(y_true, y_scores > 0.5)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Visualize true positives, false positives, and false negatives
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_scores, c=['red' if t else 'blue' for t in y_true])
plt.axhline(y=0.5, color='r', linestyle='--')
plt.xlabel('True Label')
plt.ylabel('Predicted Score')
plt.title('Classification Results')
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_2.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 3
# Generate more sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate precision and recall values
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Plot Precision-Recall curve with threshold annotations
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Thresholds')

# Annotate some threshold values
for i, thresh in enumerate(thresholds):
    if i % 100 == 0:  # Annotate every 100th threshold
        plt.annotate(f'{thresh:.2f}', (recall[i], precision[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_3.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 4
from sklearn.metrics import average_precision_score

# Calculate Average Precision
ap_score = average_precision_score(y_true, y_scores)

print(f"Average Precision Score: {ap_score:.4f}")

# Plot Precision-Recall curve with AP score
precision, recall, _ = precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {ap_score:.4f})')
plt.fill_between(recall, precision, step='post', alpha=0.2)
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_4.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 5
from sklearn.metrics import auc

# Calculate AUC
auc_score = auc(recall, precision)

print(f"Area Under the Precision-Recall Curve: {auc_score:.4f}")

# Plot Precision-Recall curve with AUC
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AUC = {auc_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with AUC')
plt.legend(loc='lower left')
plt.fill_between(recall, precision, alpha=0.2)
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_5.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 6
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Generate a binary classification dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.7, 0.3], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train different models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

plt.figure(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap_score = average_precision_score(y_test, y_scores)
    plt.plot(recall, precision, label=f'{name} (AP = {ap_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Different Models')
plt.legend(loc='lower left')
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_6.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 7
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Generate an imbalanced dataset
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model on imbalanced data
model_imb = RandomForestClassifier(random_state=42)
model_imb.fit(X_train, y_train)
y_scores_imb = model_imb.predict_proba(X_test)[:, 1]

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train model on balanced data
model_bal = RandomForestClassifier(random_state=42)
model_bal.fit(X_train_balanced, y_train_balanced)
y_scores_bal = model_bal.predict_proba(X_test)[:, 1]

# Plot Precision-Recall curves
plt.figure(figsize=(10, 6))
for y_scores, label in [(y_scores_imb, 'Imbalanced'), (y_scores_bal, 'Balanced')]:
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap_score = average_precision_score(y_test, y_scores)
    plt.plot(recall, precision, label=f'{label} (AP = {ap_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves: Imbalanced vs Balanced Data')
plt.legend(loc='lower left')
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_7.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 8
from sklearn.metrics import f1_score

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Calculate F1 score for each threshold
f1_scores = [f1_score(y_true, y_scores >= threshold) for threshold in thresholds]

# Find the threshold that gives the best F1 score
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best F1 score: {best_f1:.2f}")

# Plot Precision-Recall curve with best threshold
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.scatter(recall[np.argmax(f1_scores)], precision[np.argmax(f1_scores)], 
            color='red', label=f'Best Threshold ({best_threshold:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Best F1 Score Threshold')
plt.legend()
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_8.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 9
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample email data (content, label)
emails = [
    ("Get rich quick! Buy now!", 1),
    ("Meeting at 3pm tomorrow", 0),
    ("You've won a free iPhone!", 1),
    ("Project deadline extended", 0),
    ("Congrats! You're our lucky winner", 1),
    ("Please review the attached document", 0),
    # Add more examples...
]

# Separate content and labels
X, y = zip(*emails)

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Get predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Spam Detection')
plt.legend()
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_9.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 10
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Generate synthetic medical data
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_features=20, n_informative=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Disease Detection')
plt.legend()

# Add annotations for different thresholds
for i, thresh in enumerate(thresholds):
    if i % 100 == 0:  # Annotate every 100th threshold
        plt.annotate(f'{thresh:.2f}', (recall[i], precision[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_10.png", dpi=300, bbox_inches="tight")
plt.close()

# Print interpretation
print("In this medical diagnosis example:")
print(f"- A high precision of {precision[0]:.2f} at a low recall of {recall[0]:.2f} means we're very sure about positive diagnoses, but we might miss some cases.")
print(f"- A high recall of {recall[-1]:.2f} at a low precision of {precision[-1]:.2f} means we catch most positive cases, but with more false positives.")
print("The choice of threshold depends on the relative costs of false positives vs. false negatives in the specific medical context.")


# Code Block 11
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Generate sample data for two different scenarios
np.random.seed(42)
y_true1 = np.random.randint(0, 2, 1000)
y_scores1 = np.random.rand(1000)

y_true2 = np.random.randint(0, 2, 1000)
y_scores2 = np.random.rand(1000) * 0.5 + 0.25

# Calculate precision-recall curves
precision1, recall1, _ = precision_recall_curve(y_true1, y_scores1)
precision2, recall2, _ = precision_recall_curve(y_true2, y_scores2)

# Plot both curves
plt.figure(figsize=(10, 6))
plt.plot(recall1, precision1, label='Scenario 1')
plt.plot(recall2, precision2, label='Scenario 2')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves: Comparing Different Scenarios')
plt.legend()
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_11.png", dpi=300, bbox_inches="tight")
plt.close()

# Calculate and print average precision scores
ap1 = average_precision_score(y_true1, y_scores1)
ap2 = average_precision_score(y_true2, y_scores2)
print(f"Average Precision - Scenario 1: {ap1:.3f}")
print(f"Average Precision - Scenario 2: {ap2:.3f}")


# Code Block 12
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Generate imbalanced dataset
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.99, 0.01], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model on imbalanced data
model_imb = RandomForestClassifier(random_state=42)
model_imb.fit(X_train, y_train)
y_scores_imb = model_imb.predict_proba(X_test)[:, 1]

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train model on balanced data
model_bal = RandomForestClassifier(random_state=42)
model_bal.fit(X_train_balanced, y_train_balanced)
y_scores_bal = model_bal.predict_proba(X_test)[:, 1]

# Plot Precision-Recall curves
plt.figure(figsize=(10, 6))
for y_scores, label in [(y_scores_imb, 'Imbalanced'), (y_scores_bal, 'Balanced (SMOTE)')]:
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap_score = average_precision_score(y_test, y_scores)
    plt.plot(recall, precision, label=f'{label} (AP = {ap_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves: Imbalanced vs Balanced Data')
plt.legend()
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_12.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 13
from sklearn.metrics import roc_curve, auc

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
ap_score = average_precision_score(y_true, y_scores)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot both curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Precision-Recall curve
ax1.plot(recall, precision, label=f'AP = {ap_score:.2f}')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('Precision-Recall Curve')
ax1.legend()

# ROC curve
ax2.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], linestyle='--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend()

plt.tight_layout()
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_13.png", dpi=300, bbox_inches="tight")
plt.close()


# Code Block 14
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Generate data for a good model and a random classifier
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores_good = np.random.beta(2, 5, 1000)  # Skewed towards lower values
y_scores_random = np.random.rand(1000)

# Calculate Precision-Recall curves
precision_good, recall_good, _ = precision_recall_curve(y_true, y_scores_good)
precision_random, recall_random, _ = precision_recall_curve(y_true, y_scores_random)

# Calculate Average Precision scores
ap_score_good = average_precision_score(y_true, y_scores_good)
ap_score_random = average_precision_score(y_true, y_scores_random)

# Plot Precision-Recall curves
plt.figure(figsize=(10, 6))
plt.plot(recall_good, precision_good, label=f'Good Model (AP = {ap_score_good:.2f})')
plt.plot(recall_random, precision_random, label=f'Random Classifier (AP = {ap_score_random:.2f})')

# Add a line for the random classifier baseline
no_skill = len(y_true[y_true == 1]) / len(y_true)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves: Good Model vs Random Classifier')
plt.legend()
plt.savefig("plots/Precision-Recall%20Plots%20in%20Python_extracted_plot_14.png", dpi=300, bbox_inches="tight")
plt.close()

print("Tips for interpreting this plot:")
print("1. The 'Good Model' curve is significantly above the random classifier.")
print("2. The Average Precision (AP) score summarizes the curve's performance.")
print("3. Choose a threshold based on the trade-off between precision and recall for your specific problem.")
print("4. The 'No Skill' line represents the performance of a random classifier.")


print("\nPlots generated:")
for i, plot_file in enumerate(sorted(glob.glob('plots/*.png'))):
    print(f"{i+1}. {plot_file}")
