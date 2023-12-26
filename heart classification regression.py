import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv(r"C:\Users\HP\Desktop\machine learning\dataset\heart.csv",header=0)
print(df)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1],test_size=0.2,random_state=2)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

clf1 = LogisticRegression()

clf2 = DecisionTreeClassifier()

clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

#accuracy score

from sklearn.metrics import accuracy_score,confusion_matrix

print("Accuracy of Logistic Regression",accuracy_score(y_test,y_pred1))

print("Accuracy of Decision Trees",accuracy_score(y_test,y_pred2))

#confusion matrix

#logistic
print(confusion_matrix(y_test,y_pred1))

#decision tree
print(confusion_matrix(y_test,y_pred2))

#confusion matrix
print("Logistic Regression Confusion Matrix\n")

pd.DataFrame(confusion_matrix(y_test,y_pred1),columns=list(range(0,2)))


print("Decision Tree Confusion Matrix\n")

pd.DataFrame(confusion_matrix(y_test,y_pred2),columns=list(range(0,2)))


result = pd.DataFrame()

result['Actual Label'] = y_test

result['Logistic Regression Prediction'] = y_pred1

result['Decision Tree Prediction'] = y_pred2

print(result.sample(10))

from sklearn.metrics import recall_score,precision_score,f1_score

print("For Logistic regression Model")

print("-"*50)

cdf = pd.DataFrame(confusion_matrix(y_test,y_pred1),columns=list(range(0,2)))

print(cdf)

print("-"*50)

print("Precision - ",precision_score(y_test,y_pred1))

print("Recall - ",recall_score(y_test,y_pred1))

print("F1 score - ",f1_score(y_test,y_pred1))


print("For DT Model")

print("-"*50)

cdf = pd.DataFrame(confusion_matrix(y_test,y_pred2),columns=list(range(0,2)))

print(cdf)

print("-"*50)

print("Precision - ",precision_score(y_test,y_pred2))

print("Recall - ",recall_score(y_test,y_pred2))

print("F1 score - ",f1_score(y_test,y_pred2))