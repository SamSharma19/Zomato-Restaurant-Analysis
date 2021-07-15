import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression #Logistic Regression is a Machine Learning classification algorithm
from sklearn.linear_model import LinearRegression #Linear Regression is a Machine Learning classification algorithm
from sklearn.model_selection import train_test_split #Splitting of Dataset
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

data = pd.read_csv('C:/Users/perso/Desktop/3 year/Data project/Zomato Chennai Listing 2020.csv')
x = data.iloc[:,1:].values

#replacing all the invalid or not filled values in the dataset with NaN
data.replace(to_replace = ['None','Invalid','Does not offer Delivery','Does not offer Dining','Not enough Delivery Reviews','Not enough Dining Reviews'], value =np.nan,inplace=True)
print(data.isnull().sum())

#converting the names of restaurant to lower case
data['Name of Restaurant'] = data['Name of Restaurant'].apply(lambda x: x.lower())
data['Features'] = data["Features"].astype(str)
data['Features'] = data['Features'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))
#converting the top dishes field to string and removing its braces and '
data['Top Dishes'] = data["Top Dishes"].astype(str)
data['Top Dishes'] = data['Top Dishes'].apply(lambda x:x.replace('[','').replace(']','').replace("'",'').replace('  ','').split(','))
data['Cuisine'] = data["Cuisine"].astype(str)
data['Dining Rating'] = data['Dining Rating'].astype("Float32")
data['Dining Rating Count'] = data['Dining Rating Count'].astype("Float32")
data['Delivery Rating Count'] = data['Delivery Rating Count'].astype("Float32")
data['Delivery Rating'] = data['Delivery Rating'].astype("Float32")

def vegs(val):
    if 'Vegetarian Only' in val:
        return 'Yes'
    elif ' Vegetarian Only' in val:
        return 'Yes'
    else:
        return 'No'
data['Vegetarian Status'] = data['Features'].apply(lambda x: vegs(x))
#printing number of vegetarian only restaurants
data['Vegetarian Status'].value_counts()

zomato = data.drop(['Zomato URL', 'Top Dishes'],axis = 1)

#removing NaN values from the dataset
zomato['Dining Rating'] = zomato['Dining Rating'].fillna(data['Dining Rating'].median())
zomato['Dining Rating Count'] = zomato['Dining Rating Count'].fillna(data['Dining Rating Count'].median())
zomato['Delivery Rating'] = zomato['Delivery Rating'].fillna(data['Delivery Rating'].median())
zomato['Delivery Rating Count'] = zomato['Delivery Rating Count'].fillna(data['Delivery Rating Count'].median())


#----top 10 most popular restaurants based on the dining count rating-----------
ratcount = data.loc[data['Dining Rating'].nlargest(10).index][['Name of Restaurant','Location','Dining Rating Count']]
print(ratcount)

#----top 10 most popular restaurants based on the zomato delivery rating-----------
delcount = data.loc[data['Delivery Rating'].nlargest(10).index][['Name of Restaurant','Location','Delivery Rating Count']]
print(delcount)

#encoding input variables
Zomato =  zomato
Zomato['Name of Restaurant'] = Zomato['Name of Restaurant'].factorize()[0]
Zomato['Address'] = Zomato['Address'].factorize()[0]
Zomato['Location'] = Zomato['Location'].factorize()[0]
Zomato['Cuisine'] = Zomato['Cuisine'].factorize()[0]
Zomato['Vegetarian Status'] = Zomato['Vegetarian Status'].factorize()[0]

corr = Zomato.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
Zomato.columns

xzomato = zomato.iloc[:,[1,2,3,4,6,7,8,10]]
yzomato = zomato['Dining Rating']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(xzomato,yzomato,test_size=.1,random_state=350)
#applying linear regression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test,y_pred))

from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(xzomato,yzomato,test_size=.1,random_state=350)
DTree=DecisionTreeRegressor(min_samples_leaf=.01)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)

from sklearn.ensemble import RandomForestRegressor
RForest=RandomForestRegressor(n_estimators=80,random_state=350,min_samples_leaf=.001)
RForest.fit(x_train,y_train)
y_predict=RForest.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)

from sklearn.svm import SVR
x_train,x_test,y_train,y_test=train_test_split(xzomato,yzomato,test_size=.1,random_state=350)
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
r2_score(y_test,y_predict)

#---------------Delivery Rating -----------------------------
xzomato = zomato.iloc[:,[1,2,3,4,5,6,8]]
yzomato = zomato['Delivery Rating']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(xzomato,yzomato,test_size=.1,random_state=105)
#applying linear regression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test,y_pred))

from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(xzomato,yzomato,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)

from sklearn.ensemble import RandomForestRegressor
RForest=RandomForestRegressor(n_estimators = 150,random_state=105,min_samples_leaf=.0001)
RForest.fit(x_train,y_train)
y_predict=RForest.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)

from sklearn.svm import SVR
x_train,x_test,y_train,y_test=train_test_split(xzomato,yzomato,test_size=.1,random_state=105)
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
r2_score(y_test,y_predict)













#---------------applying classification models -----------------------
xxzomato = zomato.iloc[:,[3,4,5,6,7,8]]
yxzomato = zomato['Vegetarian Status']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(xxzomato,yxzomato,test_size = 0.1, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 10)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)
r2_score(y_test,y_predict)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)
r2_score(y_test,y_predict)


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)
r2_score(y_test,y_predict)




