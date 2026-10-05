import joblib
import numpy as np

model = joblib.load("music_genre_model.pkl")

features = np.array([[
    120,
    0.75,
    0.65,
    0.20,
    0.80,
    0.10,
    0.05
]])

prediction = model.predict(features)[0]

print("Predicted Genre:", prediction)

if hasattr(model, "predict_proba"):
    probability = model.predict_proba(features)[0]
    print("Confidence:", round(max(probability) * 100, 2), "%")
