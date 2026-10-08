
# Binary Classification Pipeline

End-to-end ML pipeline with feature engineering and ensemble methods.

---

## Installation

```bash
pip install git+https://github.com/mazyad-alrashidi/binary-classification-pipeline.git
```

---

## Usage

```python
import pandas as pd
from bcpipeline import BinaryClassifier

df = pd.read_csv('data.csv')
X = df[['Feature_1', 'Feature_2']]
y = df['Target']

clf = BinaryClassifier()
cv_results = clf.fit(X, y)
predictions = clf.predict(X)
metrics = clf.evaluate()
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

---

Made by [Mazyad Alrashidi](https://github.com/mazyad-alrashidi) | Riyadh, Saudi Arabia 🇸🇦
