# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Andra likitha
RegisterNumber: 212221220006 
*/

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
print("Placement data:")
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)#removes the specified row or col
print("Salary data:")
data1.head()

print("Checking the null() function:")
data1.isnull().sum()

print ("Data Duplicate:")
data1.duplicated().sum()

print("Print data:")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

print("Data-status value of x:")
x=data1.iloc[:,:-1]
x

print("Data-status value of y:")
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print ("y_prediction array:")
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear") #A Library for Large
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) #Accuracy Score =(TP+TN)/
#accuracy_score(y_true,y_pred,normalize=False)
print("Accuracy value:")
accuracy

from sklearn.metrics import confusion_matrix 
confusion=(y_test,y_pred) 
print("Confusion array:")
confusion

from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print("Classification report:")
print(classification_report1)

print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
![235361003-f4c121ae-ccbb-4ae8-8c8c-d2d79aaa2184](https://user-images.githubusercontent.com/131592130/236860771-3eb640e4-f6ca-47e8-8530-3adda6a7377b.png)
![235361010-5afd2cca-6663-4cd2-a7ef-28a6edf47bc6](https://user-images.githubusercontent.com/131592130/236860869-a861e800-946b-4596-9081-7e5c859d76ce.png)
![235361056-b935a3b9-3afa-4222-9ee5-9b383c3e8e8e](https://user-images.githubusercontent.com/131592130/236861035-da7d16f7-c2a2-4928-87be-9bbb82adecb5.png)
![235361056-b935a3b9-3afa-4222-9ee5-9b383c3e8e8e](https://user-images.githubusercontent.com/131592130/236861203-de838b4c-ff94-4eba-841d-9079f999f132.png)
![235361075-a20cd36b-92c8-4297-8ede-08cc69cbe801](https://user-images.githubusercontent.com/131592130/236861878-9bfecdb3-ca0a-46f0-b3c5-340b49381ce8.png)
![235361099-afa5fdd0-eb32-42b4-93f9-39804846b88b](https://user-images.githubusercontent.com/131592130/236862092-e699c0a1-0d1d-4e6b-8c59-ae00a9a28380.png)
![235361126-d240fb5b-d3a6-4226-bbd7-b8670fac7895](https://user-images.githubusercontent.com/131592130/236862309-b17a45e0-370b-4ceb-a775-26a5f6249f0b.png)
![235361147-e48f02e1-93cd-4d38-aa9f-19ca7ecb28f6](https://user-images.githubusercontent.com/131592130/236862425-be2d155a-61a3-4317-9fb9-95edd99489b2.png)
![235361183-fee3c44a-7b23-4c86-b2de-ada0389cd998](https://user-images.githubusercontent.com/131592130/236862924-a982614a-659b-4457-a14a-a1f5e29bca90.png)
![235361287-1c73101b-035d-40ec-b20f-4279387f4785](https://user-images.githubusercontent.com/131592130/236863082-a68dc82c-55a5-4e81-a96d-02ed91862a85.png)
![235361264-703784d8-19a5-4d9d-9ff2-19571a67a323](https://user-images.githubusercontent.com/131592130/236863220-892ceca8-8698-4def-b8ea-50ce16cf11f8.png)
![235361325-ff2f9e56-5ece-456b-8bf7-690d65eb97db](https://user-images.githubusercontent.com/131592130/236863308-a9628186-4ff0-407a-9ced-d89e12a7c206.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
