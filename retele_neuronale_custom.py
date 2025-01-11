import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd


# Funcții de activare
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


# Funcția de eroare
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Rețeaua Neurală
class CustomNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Inițializarea greutăților și a biasurilor
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, X):
        # Feedforward: stratul ascuns
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        # Feedforward: stratul de ieșire
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        # Calculul erorii
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Actualizarea greutăților
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X, y, epochs=1000):
        start_time = time.time()  # Start timer
        for epoch in range(epochs):
            # Pasul de antrenare
            y_pred = self.forward(X)
            self.backward(X, y)
            # Calcularea erorii
            if epoch % 100 == 0:
                loss = compute_loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        end_time = time.time()  # End timer
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

    def predict(self, X):
        # Predictia finală
        start_time = time.time()  # Start timer
        y_pred = self.forward(X)
        end_time = time.time()  # End timer
        print(f"Prediction completed in {end_time - start_time:.4f} seconds.")
        return (y_pred > 0.5).astype(int)


# 1. Load the dataset
file_path = 'online_shoppers_intention.csv'  # Schimbă cu calea ta
data = pd.read_csv(file_path)

# 2. Split the dataset into features (X) and target (y)
X = data.drop('Revenue', axis=1)
y = data['Revenue']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# 3. Preprocessing pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# 4. Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

# Convert y_resampled to numpy array for compatibility with our implementation
y_resampled = y_resampled.values.reshape(-1, 1)

# 5. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 6. Initialize and train the custom neural network
input_size = X_train.shape[1]
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 1  # Binary classification
learning_rate = 0.01

model = CustomNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
model.fit(X_train, y_train, epochs=1000)

# 7. Evaluate the model
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.2f}")
