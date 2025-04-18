import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('IFND.csv', encoding="ISO-8859-1")

# Keep only required features (only 'Statement' is used)
df = df[['Statement', 'Label']].dropna()

# Encode target variable
target_encoder = joblib.load("target_encoder.pkl") if "target_encoder.pkl" in joblib.os.listdir() else None
if target_encoder is None:
    target_encoder = joblib.load("target_encoder.pkl") if "target_encoder.pkl" in joblib.os.listdir() else None
    target_encoder = joblib.load("target_encoder.pkl") if "target_encoder.pkl" in joblib.os.listdir() else None
    target_encoder = LabelEncoder()
    df['Label'] = target_encoder.fit_transform(df['Label'])
    joblib.dump(target_encoder, "target_encoder.pkl")
else:
    df['Label'] = target_encoder.transform(df['Label'])

# Define features and target
X = df['Statement']
y = df['Label']

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Transform text data into numerical format
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")
