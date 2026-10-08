# خلية ٢: كتابة classifier.py
with open('bcpipeline/classifier.py', 'w') as f:
    f.write('''"""Binary Classification Pipeline - Reusable ML Library"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


class BinaryClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.ensemble = None
        self.feature_names = None

    def add_features(self, X):
        X_new = X.copy()
        cols = X.columns.tolist()
        if len(cols) >= 2:
            X_new["interaction"] = X[cols[0]] * X[cols[1]]
            X_new["ratio"] = X[cols[0]] / (X[cols[1]].abs() + 1e-8)
        for col in cols:
            X_new[f"{col}_sq"] = X[col] ** 2
        return X_new

    def fit(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        X_train_e = self.add_features(X_train)
        X_test_e = self.add_features(X_test)
        self.feature_names = X_train_e.columns.tolist()

        self.models = {
            "LR": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=self.random_state))]),
            "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=self.random_state))]),
            "RF": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=self.random_state))]),
            "GBM": Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(random_state=self.random_state))]),
        }

        cv_results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X_train_e, y_train, cv=5, scoring="f1")
            cv_results[name] = scores.mean()

        best_name = max(cv_results, key=cv_results.get)
        self.best_model = self.models[best_name]
        self.best_model.fit(X_train_e, y_train)

        trained = {name: m for name, m in self.models.items()}
        for name, m in trained.items():
            m.fit(X_train_e, y_train)

        self.ensemble = VotingClassifier([
            ("svm", trained["SVM"]), ("rf", trained["RF"]), ("gb", trained["GBM"])
        ], voting="soft")
        self.ensemble.fit(X_train_e, y_train)

        self.X_test_e = X_test_e
        self.y_test = y_test
        return cv_results

    def predict(self, X):
        return self.ensemble.predict(self.add_features(X))

    def evaluate(self):
        y_pred = self.ensemble.predict(self.X_test_e)
        return {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred, average="weighted"),
            "report": classification_report(self.y_test, y_pred, output_dict=True),
        }
''')

print("✅ bcpipeline/classifier.py created")