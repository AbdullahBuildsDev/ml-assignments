import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load images
IMG_SIZE = (64, 64)
data, labels = [], []
for label, folder in [(1, 'image_data/yes'), (0, 'image_data/no')]:
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            try:
                img = Image.open(os.path.join(folder, fname)).convert('L').resize(IMG_SIZE)
                data.append(np.array(img).flatten())
                labels.append(label)
            except: pass

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

results = {}

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
results['KNN'] = accuracy_score(y_test, knn_pred)
pickle.dump(knn, open('models/knn.pkl', 'wb'))
print("✅ KNN Accuracy:", results['KNN'])

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
results['Decision Tree'] = accuracy_score(y_test, dt_pred)
pickle.dump(dt, open('models/dt.pkl', 'wb'))
print("✅ Decision Tree Accuracy:", results['Decision Tree'])

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
results['Naive Bayes'] = accuracy_score(y_test, nb_pred)
pickle.dump(nb, open('models/nb.pkl', 'wb'))
print("✅ Naive Bayes Accuracy:", results['Naive Bayes'])

# Save confusion matrices
os.makedirs('static', exist_ok=True)
for name, preds in [('KNN', knn_pred), ('Decision_Tree', dt_pred), ('Naive_Bayes', nb_pred)]:
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['No Tumor','Tumor'],
                yticklabels=['No Tumor','Tumor'])
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'static/cm_{name}.png')
    plt.close()
    print(f"✅ Confusion matrix saved for {name}")

# Best model
best = max(results, key=results.get)
print(f"\n🏆 Best Model: {best} with {results[best]*100:.2f}% accuracy")
