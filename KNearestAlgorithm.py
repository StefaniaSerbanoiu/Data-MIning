import pandas as pd
import numpy as np
from collections import Counter
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('resources/online_shoppers_intention.csv')

start_time = time.time()

# Encode categorical variables
label_encoders = {}
for col in ['Month', 'VisitorType']:
    labelEncoder = LabelEncoder()
    data[col] = labelEncoder.fit_transform(data[col])
    label_encoders[col] = labelEncoder

# Normalize numerical features
scaler = MinMaxScaler()
numerical_cols = [
    'Administrative', 'Administrative_Duration', 'Informational',
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Prepare features and target
X = data.drop(columns=['Revenue']).values
y = data['Revenue'].astype(int).values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the KNearestNeighbors class
class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # The euclidian distances between the point and all training samples
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        # Find the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Initialize and train the custom KNN model
knn = KNearestNeighbors(k=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

def calculate_metrics(y_true, y_pred):
    # Initialize counts
    TP = FP = TN = FN = 0

    # Calculate TP, FP, TN, FN
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1  # True Positive
        elif true == 0 and pred == 1:
            FP += 1  # False Positive
        elif true == 0 and pred == 0:
            TN += 1  # True Negative
        elif true == 1 and pred == 0:
            FN += 1  # False Negative

    # Compute metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return accuracy, precision, recall

accuracy, precision, recall = calculate_metrics(y_test, y_pred)

print("Accuracy:", "{:.2f}".format(accuracy))
print("Precision:", "{:.2f}".format(precision))
print("Recall:", "{:.2f}".format(recall))
print("Time: %s" %"{:.2f}".format(time.time() - start_time))
