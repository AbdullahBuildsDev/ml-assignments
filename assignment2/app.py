from flask import Flask, render_template, request
import pickle, numpy as np
from PIL import Image

app = Flask(__name__)

scaler = pickle.load(open('models/scaler.pkl', 'rb'))
knn = pickle.load(open('models/knn.pkl', 'rb'))
dt = pickle.load(open('models/dt.pkl', 'rb'))
nb = pickle.load(open('models/nb.pkl', 'rb'))

def predict_all(img_file):
    img = Image.open(img_file).convert('L').resize((64,64))
    arr = np.array(img).flatten().reshape(1,-1)
    scaled = scaler.transform(arr)
    results = {}
    for name, model in [('KNN', knn), ('Decision Tree', dt), ('Naive Bayes', nb)]:
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0]
        results[name] = {
            'result': '🔴 Tumor DETECTED' if pred == 1 else '🟢 No Tumor Found',
            'confidence': f"{max(prob)*100:.1f}%",
            'color': '#e94560' if pred == 1 else '#2ecc71'
        }
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        results = predict_all(file.stream)
        return render_template('index.html', results=results)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
