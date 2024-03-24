# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('C:\\Users\\Vaidehi\\Downloads\\CreditScoringModel\\creditset.csv')  # Assuming 'credit_data.csv' contains the historical financial data


# Preprocessing
# Drop rows with missing values
data.dropna(inplace=True)

# Splitting data into features and target variable
X = data.drop('creditworthy', axis=1)
y = data['creditworthy']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100
print("Accuracy:", accuracy_percentage)

# Classification report
print (classification_report(y_test, y_pred))




import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing the distribution of the target variable 'creditworthy'
plt.figure(figsize=(8, 6))
sns.countplot(x='creditworthy', data=data)
plt.title('Distribution of Creditworthy')
plt.xlabel('Creditworthy')
plt.ylabel('Count')
plt.show()

# Visualizing feature importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Visualizing the model's performance
plt.figure(figsize=(8, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.title('Model Performance')
plt.xlabel('Observations')
plt.ylabel('Creditworthy')
plt.legend()
plt.show()
