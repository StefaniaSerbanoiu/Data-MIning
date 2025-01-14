import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# sigmoid'(z)
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


# loss function
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# neural network
class CustomNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # weights and biases for connections between input and hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        # weight and biases for connections between hidden and output layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    # processing entry data and creating a prediction
    def forward(self, X):
        # feedforward: hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        # feedforward: output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    # adjusting weights and biases to minimise the error between predictions and real values
    def backward(self, X, y):
        # error (MSE)
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X, y, epochs=1000):
        start_time = time.time()
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = compute_loss(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

    def predict(self, X):
        start_time = time.time()
        y_pred = self.forward(X)
        end_time = time.time()
        print(f"Prediction completed in {end_time - start_time:.4f} seconds.")
        return (y_pred > 0.5).astype(int)


file_path = 'online_shoppers_intention.csv'  # Schimbă cu calea ta
data = pd.read_csv(file_path)

# split the dataset into features and target
X = data.drop('Revenue', axis=1)
y = data['Revenue']

# identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_preprocessed = preprocessor.fit_transform(X)

# handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

# converting to numpy array to be useful in our algorithm
y_resampled = y_resampled.values.reshape(-1, 1)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# training
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.01

model = CustomNeuralNetwork(input_size, hidden_size, output_size, learning_rate)
model.fit(X_train, y_train, epochs=1000)

# evaluate
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
