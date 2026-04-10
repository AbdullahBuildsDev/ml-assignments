from flask import Flask, render_template, request
import pickle, numpy as np
from PIL import Image
import io, base64, os

app = Flask(__name__)

tab_model = pickle.load(open('models/tabular_model.pkl', 'rb'))
tab_scaler = pickle.load(open('models/tabular_scaler.pkl', 'rb'))
img_model = pickle.load(open('models/image_model.pkl', 'rb'))
img_scaler = pickle.load(open('models/image_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_tabular', methods=['POST'])
def predict_tabular():
    try:
        features = [float(request.form[f]) for f in ['open','high','low','close','volume']]
        scaled = tab_scaler.transform([features])
        pred = tab_model.predict(scaled)[0]
        prob = tab_model.predict_proba(scaled)[0]
        result = "📈 Price will GO UP" if pred == 1 else "📉 Price will GO DOWN"
        confidence = f"{max(prob)*100:.1f}%"
        return render_template('index.html', tab_result=result, tab_confidence=confidence)
    except Exception as e:
        return render_template('index.html', tab_result=f"Error: {e}")

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('L').resize((64,64))
        arr = np.array(img).flatten().reshape(1,-1)
        scaled = img_scaler.transform(arr)
        pred = img_model.predict(scaled)[0]
        prob = img_model.predict_proba(scaled)[0]
        result = "🔴 Tumor DETECTED" if pred == 1 else "🟢 No Tumor Found"
        confidence = f"{max(prob)*100:.1f}%"

        img_rgb = Image.open(file.stream) if False else Image.open(request.files['image'].stream)
        buffered = io.BytesIO()
        Image.open(request.files['image'].stream).save(buffered, format="JPEG")

        return render_template('index.html', img_result=result, img_confidence=confidence)
    except Exception as e:
        return render_template('index.html', img_result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
