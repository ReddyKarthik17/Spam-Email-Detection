import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv")

print("\nDataset Preview:\n")
print(data.head())

# Features and labels
X = data["Message"]
y = data["Label"]

# Convert text into numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# User input
user_message = input("\nEnter a message: ")

# Transform input
message_vector = vectorizer.transform([user_message])

# Predict
result = model.predict(message_vector)

print(f"\nPrediction: {result[0]}")
