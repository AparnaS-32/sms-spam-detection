
import pandas as pd

# Load the SMS dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display the first five messages
print("SMS SPAM DETECTION")
print("------------------")

print("\nFirst 5 Messages:")
print(data.head())

# Display dataset information
print("\nDataset Shape:")
print(data.shape)

# Count spam and normal messages
print("\nMessage Distribution:")
print(data["label"].value_counts())

from sklearn.model_selection import train_test_split

# Separate messages and labels
X = data["message"]
y = data["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nDATASET SPLITTING")
print("-----------------")
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text messages into numerical features
vectorizer = TfidfVectorizer(stop_words="english")

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

print("\nTEXT VECTORIZATION")
print("------------------")
print("Training features shape:", X_train_features.shape)
print("Testing features shape:", X_test_features.shape)
from sklearn.naive_bayes import MultinomialNB

# Train the spam detection model
model = MultinomialNB()
model.fit(X_train_features, y_train)

print("\nMODEL TRAINING")
print("--------------")
print("Model training completed successfully!")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict test messages
y_pred = model.predict(X_test_features)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("----------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Test custom SMS messages
test_messages = [
    "Congratulations! You won a free iPhone. Click here to claim.",
    "Hey, are we meeting tomorrow?",
    "URGENT! You have won a cash prize. Call now!"
]

# Convert messages into numerical features
test_features = vectorizer.transform(test_messages)

# Predict message categories
predictions = model.predict(test_features)

print("\nCUSTOM MESSAGE PREDICTIONS")
print("--------------------------")

for message, prediction in zip(test_messages, predictions):
    print("\nMessage:", message)
    print("Prediction:", prediction.upper())
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Ham", "Spam"],
    cmap="Blues"
)

plt.title("SMS Spam Detection - Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()