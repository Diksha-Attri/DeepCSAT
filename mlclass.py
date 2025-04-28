# ==========================================
# DeepCSAT - Classification Model Version
# ==========================================

# Step 1: Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

# Step 2: Load the Dataset
print("\n📥 Loading the Dataset...")
data = pd.read_csv(r'C:\Users\Diksha\OneDrive\Desktop\Diksha Python\DeepCSAT\dataset\eCommerce_Customer_support_data.csv')

# Step 3: Preprocessing the Data
print("\n🛠️ Preprocessing Data...")

# Drop useless columns
useless_cols = ['Unique id', 'Order_id', 'order_date_time', 'Issue_reported at', 'issue_responded', 'Survey_response_Date', 'Customer Remarks']
data.drop(columns=useless_cols, inplace=True)

# Convert important numeric columns
data['Item_price'] = pd.to_numeric(data['Item_price'], errors='coerce')
data['connected_handling_time'] = pd.to_numeric(data['connected_handling_time'], errors='coerce')

# Fill missing numeric values
data['Item_price'].fillna(data['Item_price'].median(), inplace=True)
data['connected_handling_time'].fillna(data['connected_handling_time'].median(), inplace=True)

# Encode categorical columns
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Step 4: Convert CSAT Scores into Categories (Classes)
print("\n🎯 Converting CSAT Scores into Categories...")

def convert_csat(score):
    if score <= 2:
        return 0   # Dissatisfied
    elif score == 3:
        return 1   # Neutral
    else:
        return 2   # Satisfied

data['CSAT Category'] = data['CSAT Score'].apply(convert_csat)




# Step 5: Feature and Target Split
feature_cols = ['channel_name', 'category', 'Sub-category', 'Customer_City', 'Product_category',
                'Item_price', 'connected_handling_time', 'Agent_name', 'Supervisor',
                'Manager', 'Tenure Bucket', 'Agent Shift']

X = data[feature_cols]
y = data['CSAT Category']

# Step 6: Train-Test Split
print("\n✂️ Splitting Data into Train and Test sets (80%-20%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build the Random Forest Classifier
print("\n🌲 Building Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

# Step 8: Train the Model
print("\n🚀 Training the Random Forest Classifier...")
rf_model.fit(X_train, y_train)

# Step 9: Make Predictions
print("\n🔮 Making Predictions on Test Set...")
y_pred = rf_model.predict(X_test)

# Step 10: Model Evaluation
print("\n📊 Evaluating the Model...")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dissatisfied', 'Neutral', 'Satisfied']))

# Step 11: Confusion Matrix Plot
print("\n📈 Plotting Confusion Matrix...")
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Dissatisfied', 'Neutral', 'Satisfied'],
            yticklabels=['Dissatisfied', 'Neutral', 'Satisfied'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - CSAT Classification')
plt.show()

# Step 12: Save the Classifier Model
print("\n💾 Saving the Model...")
if not os.path.exists('model'):
    os.makedirs('model')

import joblib
joblib.dump(rf_model, 'model/deepcsat_classifier.pkl')
print("✅ Classifier model saved as 'deepcsat_classifier.pkl' inside model/ folder.")

# Step 13: Conclusion
print("\n✅ Classification Model Completed Successfully!")
