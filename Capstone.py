import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data=pd.read_csv('D:\ML\employeedata.csv')

print(data.head())
print(data.info())
print(data.describe())

print(data.isnull().sum())

#sns.pairplot(data,hue='left')
#plt.show()
left_data=data[data['left']==1]
#sns.scatterplot(data=left_data,x='satisfaction_level',y='average_montly_hours')
#plt.title('Satisfaction Level vs Average Monthly Hours for Employees who Left')
#plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=left_data, x='Department', y='satisfaction_level')
plt.title('Satisfaction Level by Department for Employees Who Left')
plt.xticks(rotation=45)
plt.show()

# Promotion in last 5 years vs Satisfaction Level
sns.catplot(data=left_data, x='promotion_last_5years', y='satisfaction_level', kind='box')
plt.title('Satisfaction Level by Promotion in Last 5 Years for Employees Who Left')
plt.show()

# Salary level vs Satisfaction Level
sns.catplot(data=left_data, x='salary', y='satisfaction_level', kind='box')
plt.title('Satisfaction Level by Salary Level for Employees Who Left')
plt.show()

#Data Preprocessing
X=data.drop('left', axis=1)
y=data['left']

categorical_features=['Department', 'salary']
numerical_features=X.drop(categorical_features, axis=1).columns
prepocessor=ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])


model= Pipeline(steps=[ ('preprocessor', prepocessor),('classifier', RandomForestClassifier(random_state=42))
                       ])

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print('Cross-validation scores:', cv_scores)
print('Mean cross-validation score:', np.mean(cv_scores))






