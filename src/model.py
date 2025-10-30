# src/model.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess_data


# -----------------------------------
# 🔹 1. Load + Preprocess the Dataset
# -----------------------------------
import pandas as pd
df = pd.read_csv("data/internships.csv")

X_train, X_test, y_train, y_test, tfidf = preprocess_data(df)


# -----------------------------------
# 🔹 2. Handle Class Imbalance (SMOTE)
# -----------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("✅ After SMOTE:", np.bincount(y_train_res))


# -----------------------------------
# 🔹 3. Build the Random Forest Model
# -----------------------------------
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# Define hyperparameter grid for tuning
param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    rf, 
    param_distributions=param_dist, 
    n_iter=10, 
    scoring='f1',
    cv=5, 
    verbose=2, 
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_res, y_train_res)

best_rf = random_search.best_estimator_
print("✅ Best Parameters Found:", random_search.best_params_)


# -----------------------------------
# 🔹 4. Evaluate the Model
# -----------------------------------
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# -----------------------------------
# 🔹 5. Save Model + Vectorizer
# -----------------------------------
joblib.dump(best_rf, "model/random_forest_model.joblib")
joblib.dump(tfidf, "model/tfidf_vectorizer.joblib")

print("✅ Model and TF-IDF Vectorizer saved successfully!")
