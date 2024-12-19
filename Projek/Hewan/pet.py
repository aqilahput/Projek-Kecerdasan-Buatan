import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv(r'D:/UAS KB/Hewan/data.csv')

# Fill missing values in the target column
data['Dangerous'] = data['Dangerous'].fillna('Unknown')

# Combine symptom columns into a single text column to represent activity
data['Activity'] = data[['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']].agg(' '.join, axis=1)

# Encode the target column into binary values (Yes: 1, No: 0)
data = data[data['Dangerous'] != 'Unknown']  # Exclude rows with 'Unknown'
data['Dangerous'] = data['Dangerous'].map({'Yes': 1, 'No': 0})

# Split data into features and target
X = data['Activity']
y = data['Dangerous']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: Convert text to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_vec, y_train)

# Make predictions
y_pred = rf_model.predict(X_val_vec)

# Evaluate the model
print("Classification Report:")
classification_report_text = classification_report(y_val, y_pred, target_names=['Not Dangerous', 'Dangerous'])
print(classification_report_text)

# Save classification report to file
with open('classification_report.txt', 'w') as file:
    file.write(classification_report_text)

conf_matrix = confusion_matrix(y_val, y_pred)

# Calculate accuracy percentage
accuracy_percent = np.diag(conf_matrix).sum() / conf_matrix.sum() * 100

# Save accuracy as percentage in a text file
with open('accuracy_percentage.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy_percent:.2f}%')

# Plot confusion matrix with accuracy percentage
def plot_confusion_matrix(conf_matrix, classes, filename='confusion_matrix.png'):
    plt.figure(figsize=(6, 6))
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(conf_matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Annotate the matrix with counts and accuracy percentage
    thresh = conf_matrix_normalized.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, f"{conf_matrix[i, j]} ({conf_matrix_normalized[i, j] * 100:.2f}%)",
                 horizontalalignment="center",
                 color="white" if conf_matrix_normalized[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)  # Save plot to file
    plt.show()

plot_confusion_matrix(conf_matrix, ['Not Dangerous', 'Dangerous'])

# Load saved plot if needed
from PIL import Image
image = Image.open('confusion_matrix.png')
image.show()
