from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
MODEL_PATH = "./hate_speech_distilbert"  # Update with actual path
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# Example Usage
if __name__ == "__main__":
    sample_text ="Let's GOOO, we did it!"
    result = predict(sample_text)
    print(f"Prediction: {result}")
