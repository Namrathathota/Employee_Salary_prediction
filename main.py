import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Load dataset
data = pd.read_csv('employee salary prediction dataset.csv')

#  Display basic dataset info
print(" Dataset Loaded Successfully")
print("\nFirst 5 rows:")
print(data.head())
print("\nColumns:", data.columns)

#  Target column
target_col = "income"
print(f"\n Target Column Detected: {target_col}")

#  Features (X) and Target (y)
X = data.drop(target_col, axis=1)
y = data[target_col]

#  Encode target labels ('<=50K' -> 0, '>50K' -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nTarget Classes Encoded:")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("\n Categorical Columns:", categorical_cols)
print(" Numerical Columns:", numerical_cols)

#  Preprocessing: Scale numeric + encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Random Forest Classifier with 200 trees
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

#  Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

#  Train the model
print("\n Training model...")
model.fit(X_train, y_train)

#  Predict
y_pred = model.predict(X_test)

#  Evaluate performance
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n Model Accuracy: {accuracy:.2f}%")

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
