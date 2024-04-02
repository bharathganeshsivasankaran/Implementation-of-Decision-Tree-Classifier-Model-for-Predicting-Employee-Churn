# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas module and import the required data set.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Bharathganesh S
RegisterNumber:  212222230022
*/
```
```
import pandas as pd
data=pd.read_csv('/content/Employee.csv')
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
### Displaying the head of the dataset
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119478098/26f4f1ae-5007-4010-880b-783e865d95a1)
### Showing the information about the dataset
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119478098/d60e5f0a-1c84-4b3f-984b-71c0b2bd02ad)
### Printing null values in the dataset
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119478098/27728d8e-7086-4594-a074-f5e6346b07ef)
### Value counts of the 'left' column
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119478098/2f5bf735-b2ac-4157-a97f-69de7891a1a8)
### Label encoding the values of the salary column
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119478098/ee05d00a-eb8f-498f-84ac-c68c7bf5db92)
### Spliting the columns for getting input and output
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119478098/de8b1989-b02d-4eeb-9bc8-e45ada87f318)
### Creating a Decision Tree Classifier
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119478098/f49017a4-84d1-478b-9436-588d290efd27)
### Finding the accuracy for the test data
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119478098/c4964325-774c-4ee5-8dcd-03bb81031b0e)
### Testing the model
![image](https://github.com/bharathganeshsivasankaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119478098/3ef6abd4-ae1e-4e59-a3f3-2e95491a76b4)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
