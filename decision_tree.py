import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Load dataset
df = pd.read_csv("C:/Users/STEFANA DCRUZ/Downloads/heart.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(dtree, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree")
plt.show()

# 2. Overfitting Analysis
depths = range(1, 21)
train_scores, test_scores = [], []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, label="Training Accuracy")
plt.plot(depths, test_scores, label="Testing Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Overfitting Analysis by Tree Depth")
plt.grid(True)
plt.show()

# 3. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Accuracy comparison
print("Decision Tree Accuracy:", accuracy_score(y_test, dtree.predict(X_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

# 4. Feature Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Feature Importances (Random Forest)")
plt.show()

# 5. Cross-Validation
cv_dt = cross_val_score(dtree, X, y, cv=5).mean()
cv_rf = cross_val_score(rf, X, y, cv=5).mean()
print("Cross-Validation Accuracy (Decision Tree):", cv_dt)
print("Cross-Validation Accuracy (Random Forest):", cv_rf)

# Classification report
print("\nClassification Report (Random Forest):\n")
print(classification_report(y_test, rf_preds))

# 6. Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("Best Random Forest Parameters:", grid_search.best_params_)

# 7. Confusion Matrix
best_rf_preds = best_rf.predict(X_test)
cm = confusion_matrix(y_test, best_rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Best Random Forest)")
plt.show()

# 8. ROC Curve and AUC
y_proba = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
