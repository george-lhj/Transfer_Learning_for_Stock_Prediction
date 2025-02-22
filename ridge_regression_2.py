import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data loading
df = pd.read_csv("signature.csv")
X_train = pd.DataFrame(df)
feature_names = [[f"SIG_{i}" for i in range(40)]]
y_train_df = pd.read_csv(r"C:\Users\shirl\OneDrive\桌面\ucb capstone qrt\raw data provided by qrt\y_train.csv")  # Load target variable

# Target variable processing
train_ids = X_train["ID"].astype(int)  # Extract stock IDs
y_train_df["RET"] = (y_train_df["RET"] > 0).astype(int)  # Convert returns to binary labels
y_dict = dict(zip(y_train_df["ID"], y_train_df["RET"]))  # Create ID-to-label mapping
y_train = np.array([y_dict.get(stock_id, 0) for stock_id in train_ids])  # Generate training labels

# Feature processing
# Remove non-feature columns
cols_to_drop = ["ID", "Stock"]
X_train.drop(columns=cols_to_drop, inplace=True, errors="ignore")  # Safely remove specified columns

# Standardization
scaler = StandardScaler()  # Create scaler
X_scaled = scaler.fit_transform(X_train)  # Perform standardization

# # Data splitting
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_scaled, y_train, 
    test_size=0.2,  # Validation set ratio
    random_state=42  # Random seed
)

# Model training
ridge = RidgeClassifier(
    alpha=2.15443469  # Regularization coefficient
)
ridge.fit(X_train_sub, y_train_sub)  # Train model '''x_train, y_train'''

# Validation prediction
y_val_pred = ridge.predict(X_val)  # Generate predictions '''y_test = predict(X_test)'''

# Result output
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")  # Print accuracy
print("Classification Report:")
print(classification_report(y_val, y_val_pred))  # Output detailed metrics

# validation accuracy: 0.5135