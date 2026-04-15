import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# 1. IMPORT XGBOOST THAY CHO RANDOM FOREST
from xgboost import XGBClassifier 

# (Giữ nguyên phần định nghĩa đường dẫn và load data của bạn)
data_path = '/home/mikey/Lab/project_t2_hk1/PredictSecBugs/aprrentissage/result_total_bon.csv'
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
# Lưu ý: Ở Pandas mới, cách đổi datetime này có thể cần dùngastype('int64') thay vì view()
X['commit_date'] = pd.to_datetime(X['commit_date']).astype('int64') / 10**9

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

# 2. KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH XGBOOST
# eval_metric='logloss' giúp tránh cảnh báo (warning) ở các phiên bản XGBoost mới
classifier = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=6, 
    random_state=42, 
    eval_metric='logloss'
)
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
print("Feature Importances:\n", importances_df.head(10)) # Hiển thị 10 đặc trưng quan trọng nhất
