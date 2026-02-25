#?------------------------------------------------------------
#? Author: Nolan Tuttle
#? Project: Iris
#?------------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

iris = load_iris()

x = iris.data # input data (features)
y = iris.target # output data

feature_names = iris.feature_names # feature names
target_names = iris.target_names # target names

print("Features: ", feature_names)
print("Target Names: ", target_names)
print("First 5 Samples: ", x[:5])

# Data preprocessing step 1: Check for missing, incomplete, or duplicate data
data = pd.DataFrame(iris.data, columns=feature_names)

data.drop_duplicates(inplace=True) # Remove duplicate rows

print("\nMissing value per feature: ", data.isnull().sum()) # Check for missing values
print("\nDuplicate rows: ", data.duplicated().sum()) # Check for duplicate 


# Data preprocessing step 2: Standardize feature values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("\nFirst 5 Scaled Training Data: ", x_train_scaled[:5])

# Step 3: Train and evaluate a logistic regression model

model = LogisticRegression()
model.fit(x_train_scaled, y_train);

# Save the trained model and scaler to a file
with open("model_and_scaler.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

y_pred = model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix is used to evaluate performance of a classification model
# They breakdown predictions compared to actual outcomes
cm = confusion_matrix(y_test, y_pred)

# Visualize Confusion Matrix using seaborn heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()