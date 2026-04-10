from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

print("⏳ Loading model...")
sentiment = pipeline("sentiment-analysis", model="model/")
print("✅ Model ready!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    result = sentiment(text)[0]
    label = result['label']
    score = f"{result['score']*100:.1f}%"
    emoji = "😊 POSITIVE" if label == "POSITIVE" else "😞 NEGATIVE"
    color = "#2ecc71" if label == "POSITIVE" else "#e94560"
    return render_template('index.html', result=emoji, confidence=score, color=color, text=text)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
