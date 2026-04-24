

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    'age': [25, 32, 47, 51, 62, 23, 40, 36, 28, 55],
    'salary': [20000, 30000, 45000, 50000, 65000, 18000, 40000, 38000, 22000, 52000],
    'previous_purchase': [0,1,1,1,0,0,1,0,0,1],
    'subscribed': [0,1,1,1,0,0,1,0,0,1]
}

df = pd.DataFrame(data)

X = df[['age','salary','previous_purchase']]
y = df['subscribed']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = DecisionTreeClassifier()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))


print(classification_report(y_test, y_pred))
