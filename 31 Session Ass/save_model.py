import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("music_features.csv")

X = data.drop("genre", axis=1)
y = data["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "music_genre_model.pkl")

print("Model saved as music_genre_model.pkl")
print("Accuracy:", model.score(X_test, y_test))
