import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load data
df = pd.read_csv('tabular_data/finalgolddata.csv')

# Create binary label: 1 = price went UP, 0 = price went DOWN
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# Features
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Tabular Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and scaler
pickle.dump(model, open('models/tabular_model.pkl', 'wb'))
pickle.dump(scaler, open('models/tabular_scaler.pkl', 'wb'))
print("✅ Model saved to models/tabular_model.pkl")
