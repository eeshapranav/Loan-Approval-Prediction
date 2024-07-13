import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# Read CSV file
data = pd.read_csv(r'C:\Users\eesha\OneDrive\Documents\LoanApprovalPrediction.csv')
print(data.head(5))

# Get the number of object type columns
obj = (data.dtypes == 'object')
print("Categorical variables:", len(list(obj[obj].index)))

# Visualize the categorical columns using countplot
object_cols = list(obj[obj].index)
plt.figure(figsize=(18, 36))
index = 1

for col in object_cols:
    plt.subplot(4, 2, index)
    sns.countplot(x=col, data=data)
    plt.title(col)
    plt.xticks(rotation=90)
    index += 1

plt.tight_layout()
plt.show()

# Use Label Encoder to encode the categorical columns
label_encoder = preprocessing.LabelEncoder()
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

# Encode the target variable 'Loan_Status' if not already encoded
if 'Loan_Status' in data.columns:
    data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])

print("Label Encoder:", label_encoder)

# Find the number of object type columns
obj = (data.dtypes == 'object')
print("Categorical variables:", len(list(obj[obj].index)))

plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

sns.catplot(x="Gender", y="Married", hue="Loan_Status", kind="bar", data=data)
plt.show()

# Handle missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())

print(data.isna().sum())

# Split the data into training and testing sets
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']
print(X.shape, Y.shape)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.4, random_state=1)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Initialize classifiers
knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
svc = SVC()
lc = LogisticRegression(max_iter=500)  # Increase the number of iterations

# Handle convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Make predictions on the training set
for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    print(f"Accuracy score of {clf.__class__.__name__} = {100 * metrics.accuracy_score(Y_train, Y_pred):.2f}%")

# Evaluate on the test set
for clf in (rfc, knn, svc, lc):
    Y_pred_test = clf.predict(X_test)
    print(f"Test Accuracy score of {clf.__class__.__name__} = {100 * metrics.accuracy_score(Y_test, Y_pred_test):.2f}%")
