#!/usr/bin/env python
# coding: utf-8

# In[6]:




# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[29]:


import pandas as pd
resultat=pd.read_csv('/home/mikey/Lab/project_t2_hk1/PredictSecBugs/aprrentissage/result_total_bon.csv')


# In[30]:


resultat['buggy'].value_counts()


# In[31]:


resultat[resultat['buggy']==1]


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Exemple d'un DataFrame
bugged_files = pd.read_csv('/home/mikey/Lab/project_t2_hk1/PredictSecBugs/aprrentissage/result_total_bon.csv')


# In[33]:


bugged_files.shape


# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


distribution = bugged_files.groupby(['Ecosystem', 'extension', 'Source']).size().unstack()


distribution.plot(kind='bar', stacked=True)
plt.title('Distribution des Fichiers Avant et Après Correction par Ecosystem et Extension')
plt.xlabel('Ecosystem et Extension')
plt.ylabel('Nombre de Fichiers')
plt.show()


# In[35]:


sns.countplot(x='Ecosystem', hue='Source', data=bugged_files)
plt.title('Distribution des Fichiers Avant et Après Correction par Ecosystem')
plt.xlabel('Ecosystem')
plt.ylabel('Nombre de Fichiers')
plt.show()


# In[36]:


sns.countplot(x='extension', hue='Source', data=bugged_files)
plt.title('Distribution des Fichiers Avant et Après Correction par Extension')
plt.xlabel('Extension')
plt.ylabel('Nombre de Fichiers')
plt.show()


# In[37]:


contingency_table = pd.crosstab([bugged_files['Ecosystem'], bugged_files['extension']], bugged_files['Source'])

sns.heatmap(contingency_table, annot=True, cmap="YlGnBu")
plt.title('Heatmap des Fichiers Avant et Après Correction par Ecosystem et Extension')
plt.show()


# In[38]:


bugged_files['commit_date'] = pd.to_datetime(bugged_files['commit_date'])


# In[39]:


bugged_files


# In[40]:


bugged_files[bugged_files['buggy']==1].shape


# In[ ]:




# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path, selected_columns, target_column):
    # Load the dataset
    data = pd.read_csv(file_path)
    data = data.sort_values(by='commit_date')  # Sort by commit date

    # Select features and target
    X = data[selected_columns]
    y = data[target_column]

    # Convert 'commit_date' to numerical features
    X = X.copy()
    # Chuyển đổi sang kiểu số trước, sau đó gán lại hoàn toàn để Pandas đổi kiểu dữ liệu của cột
    commit_date_numeric = pd.to_datetime(X['commit_date']).values.view('int64') / 10**9
    X['commit_date'] = commit_date_numeric

    return X, y

# Function to create the preprocessor
def create_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols),
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_cols)
        ]
    )
    return preprocessor

# Function to apply SMOTE
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Main function to train and evaluate the model
def train_and_evaluate_model(file_path, selected_columns, target_column, model: BaseEstimator):
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path, selected_columns, target_column)

    # Create preprocessor
    preprocessor = create_preprocessor(X)

    # Preprocess data
    X_preprocessed = preprocessor.fit_transform(X)

    # Apply SMOTE to oversample the minority class
    X_resampled, y_resampled = apply_smote(X_preprocessed, y)

    # Initialize TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform cross-validation
    scores = cross_val_score(model, X_resampled, y_resampled, cv=tscv, scoring='accuracy')
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", np.mean(scores))

    # Train-test split for final evaluation
    train_size = int(0.7 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Preprocess training and test sets
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Apply SMOTE to training data
    X_train_resampled, y_train_resampled = apply_smote(X_train_preprocessed, y_train)

    # Train the model
    model.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred = model.predict(X_test_preprocessed)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Final Test Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Example usage
if __name__ == "__main__":
    # File path and columns
    file_path = '/home/mikey/Lab/project_t2_hk1/PredictSecBugs/aprrentissage/result_total_bon.csv'
    selected_columns = ['commit_date', 'extension', 'AvgLineCode', 'CountDeclClass', 'RatioCommentToCode',
                        'CountStmtExe', 'AvgCyclomaticStrict', 'CountLine', 'SumCyclomatic',
                        'AvgCyclomatic', 'SumEssential', 'MaxCyclomatic', 'AvgLineComment',
                        'AvgCyclomaticModified', 'AvgEssential', 'SumCyclomaticModified',
                        'CountLineComment', 'CountLineCode', 'MaxCyclomaticModified',
                        'CountLineBlank', 'CountStmtDecl', 'AvgLine', 'MaxEssential',
                        'CountDeclFunction', 'MaxNesting', 'AvgLineBlank',
                        'SumCyclomaticStrict', 'CountStmt', 'Ecosystem']
    target_column = 'buggy'

    # Choose your model
    chosen_model = LogisticRegression(random_state=42, max_iter=1000)

    # Train and evaluate the model
    train_and_evaluate_model(file_path, selected_columns, target_column, chosen_model)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Load the dataset and sort it by the commit date
data_path = 'Selection deleted'
# Function to create the preprocessor
def create_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols),
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_cols)
        ]
    )
    return preprocessor

# Function to apply SMOTE
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Main function to train and evaluate the model
def train_and_evaluate_model(file_path, selected_columns, target_column, model: BaseEstimator):
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path, selected_columns, target_column)

    # Create preprocessor
    preprocessor = create_preprocessor(X)

    # Preprocess data
    X_preprocessed = preprocessor.fit_transform(X)

    # Apply SMOTE to oversample the minority class
    X_resampled, y_resampled = apply_smote(X_preprocessed, y)

    # Initialize TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform cross-validation
    scores = cross_val_score(model, X_resampled, y_resampled, cv=tscv, scoring='accuracy')
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", np.mean(scores))

    # Train-test split for final evaluation
    train_size = int(0.7 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Preprocess training and test sets
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Apply SMOTE to training data
    X_train_resampled, y_train_resampled = apply_smote(X_train_preprocessed, y_train)

    # Train the model
    model.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred = model.predict(X_test_preprocessed)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Final Test Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Example usage
if __name__ == "__main__":
    # File path and columns
    file_path = '/home/mikey/Lab/project_t2_hk1/PredictSecBugs/aprrentissage/result_total_bon.csv'
    selected_columns = ['commit_date', 'extension', 'AvgLineCode', 'CountDeclClass', 'RatioCommentToCode',
                        'CountStmtExe', 'AvgCyclomaticStrict', 'CountLine', 'SumCyclomatic',
                        'AvgCyclomatic', 'SumEssential', 'MaxCyclomatic', 'AvgLineComment',
                        'AvgCyclomaticModified', 'AvgEssential', 'SumCyclomaticModified',
                        'CountLineComment', 'CountLineCode', 'MaxCyclomaticModified',
                        'CountLineBlank', 'CountStmtDecl', 'AvgLine', 'MaxEssential',
                        'CountDeclFunction', 'MaxNesting', 'AvgLineBlank',
                        'SumCyclomaticStrict', 'CountStmt', 'Ecosystem']
    target_column = 'buggy'

    # Choose your model
    chosen_model = LogisticRegression(random_state=42, max_iter=1000)

    # Train and evaluate the model
    train_and_evaluate_model(file_path, selected_columns, target_column, chosen_model)

data = pd.read_csv(data_path)
data = data.sort_values(by='commit_date')

# Specify the columns to be used for training
selected_columns = ['commit_date', 'extension', 'AvgLineCode', 'CountDeclClass', 'RatioCommentToCode',
                    'CountStmtExe', 'AvgCyclomaticStrict', 'CountLine', 'SumCyclomatic', 'AvgCyclomatic',
                    'SumEssential', 'MaxCyclomatic', 'AvgLineComment', 'AvgCyclomaticModified', 'AvgEssential',
                    'SumCyclomaticModified', 'CountLineComment', 'CountLineCode', 'MaxCyclomaticModified',
                    'CountLineBlank', 'CountStmtDecl', 'AvgLine', 'MaxEssential', 'CountDeclFunction',
                    'MaxNesting', 'AvgLineBlank', 'SumCyclomaticStrict', 'CountStmt', 'Ecosystem']
target_column = 'buggy'

# Filter the DataFrame to keep only the selected columns
X = data[selected_columns]
y = data[target_column]

# Convert 'commit_date' to numerical features (Unix timestamp)
X = X.copy()
X['commit_date'] = pd.to_datetime(X['commit_date']).view(np.int64) / 10**9

# Split the data into training and test sets chronologically
train_size = int(0.7 * len(X))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Identify categorical columns for encoding
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Create a preprocessor for handling missing values and encoding features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols),
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())  # Normalize numerical features
        ]), X_train.select_dtypes(include=['float64', 'int64']).columns)
    ]
)

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Train a Random Forest model (can be replaced with any classifier)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)

# Preprocess the test data
X_test_preprocessed = preprocessor.transform(X_test)

# Make predictions on the test set
y_pred = classifier.predict(X_test_preprocessed)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Retrieve feature names after preprocessing
encoded_categorical_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
feature_names = list(encoded_categorical_names) + list(numerical_cols)

# Calculate feature importance
feature_importances = classifier.feature_importances_

# Create a DataFrame to display feature importance
importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort features by importance
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Display the feature importance DataFrame
print("Feature Importances:\n", importances_df)


# In[ ]:





# In[ ]:




