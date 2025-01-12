import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import datetime
from sklearn.metrics import precision_score, recall_score

# Record start time
timestamp = datetime.datetime.now()

# Helper functions
def gini_index(groups, classes):
    """ Calculate the Gini Index for a split dataset """
    n_instances = float(sum(len(group) for group in groups))     
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        outcomes = np.array([row[-1] for row in group])
        for class_val in classes:
            p = np.sum(outcomes == class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def test_split(index, value, dataset):
    """ Split a dataset based on an attribute and an attribute value """
    left = [row for row in dataset if row[index] < value]
    right = [row for row in dataset if row[index] >= value]
    return left, right

def get_split(dataset, n_features):
    """ Select the best split point for a dataset with random feature selection """
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    features = np.random.choice(range(len(dataset[0]) - 1), n_features, replace=False)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def to_terminal(group):
    """ Create a terminal node value """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth, n_features):
    """ Create child splits for a node or make terminal """
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, depth + 1, n_features)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, depth + 1, n_features)

def build_tree(train, max_depth, min_size, n_features):
    """ Build a decision tree """
    root = get_split(train, n_features)
    split(root, max_depth, min_size, 1, n_features)
    return root

def predict(node, row):
    """ Make a prediction with a decision tree """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def subsample(dataset, ratio):
    """ Create a random subsample from the dataset with replacement """
    n_sample = round(len(dataset) * ratio)
    indices = np.random.choice(len(dataset), n_sample, replace=True)
    return [dataset[i] for i in indices]

def bagging_predict(trees, row):
    """ Make a prediction with a list of bagged trees """
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def random_forest_parallel(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    """ Random Forest with Parallel Tree Building """
    def build_and_store_tree(_):
        sample = subsample(train, sample_size)
        return build_tree(sample, max_depth, min_size, n_features)

    # Build trees in parallel
    trees = Parallel(n_jobs=-1)(delayed(build_and_store_tree)(_) for _ in range(n_trees))
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions

# Load the dataset
file_path = 'online_shoppers_intention.csv'  # Replace with your path
data = pd.read_csv(file_path)
print(timestamp)

# Split the dataset into features (X) and target (y)
X = data.drop('Revenue', axis=1)
y = data['Revenue']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# Preprocessing pipeline
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

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

# Combine X and y for custom implementation
dataset = np.hstack((X_resampled, y_resampled.values.reshape(-1, 1)))

# Split dataset into train and test sets
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# Parameters for Random Forest
max_depth = 10
min_size = 10
sample_size = 0.8
n_trees = 5  # Reduced for optimization
n_features = int(np.sqrt(len(train[0]) - 1))  # Number of features for each split

# Run Random Forest
predictions = random_forest_parallel(train, test, max_depth, min_size, sample_size, n_trees, n_features)

# Evaluate the model
actual = [row[-1] for row in test]
accuracy = sum(1 for i in range(len(actual)) if actual[i] == predictions[i]) / len(actual)

# Convert predictions and actual values to NumPy arrays
predictions = np.array(predictions)
actual = np.array(actual)

# Calculate Precision and Recall
precision = precision_score(actual, predictions)
recall = recall_score(actual, predictions)

# Display the results
timestamp1 = datetime.datetime.now()
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Time: {timestamp1}")
