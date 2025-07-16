import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the cleaned CSV with 'stress_level' column
df = pd.read_csv('cleaned_stress_data.csv')

# Split into features and target
X = df.drop('anxiety_(scale_1–5)', axis=1)
y = df['anxiety_(scale_1–5)']


# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'mental_stress_detector.pkl')

print("✅ Model trained and saved as mental_stress_detector.pkl")
