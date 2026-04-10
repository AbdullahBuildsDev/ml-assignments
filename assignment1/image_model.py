import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

IMG_SIZE = (64, 64)
data, labels = [], []

for label, folder in [(1, 'image_data/yes'), (0, 'image_data/no')]:
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = Image.open(os.path.join(folder, fname)).convert('L').resize(IMG_SIZE)
                data.append(np.array(img).flatten())
                labels.append(label)
            except:
                pass

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Image Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pickle.dump(model, open('models/image_model.pkl', 'wb'))
pickle.dump(scaler, open('models/image_scaler.pkl', 'wb'))
print("✅ Image model saved!")
