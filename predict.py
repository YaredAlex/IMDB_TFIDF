import sys
from classifier.predict import predict_sentiment

if len(sys.argv) < 2:
    print("Usage: python predict.py \"Your review text here\"")
    sys.exit()

review_text = sys.argv[1]
label, confidence = predict_sentiment(review_text)

print(f"Prediction: {label} (confidence: {confidence:.2f})")
