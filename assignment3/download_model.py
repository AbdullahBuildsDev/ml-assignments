from transformers import pipeline

print("⏳ Downloading model from HuggingFace...")
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
sentiment.save_pretrained("model/")
print("✅ Model downloaded and saved to model/ folder!")
