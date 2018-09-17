from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pandas as pd
from time import time
from IPython.display import display


import visuals as vs
data = pd.read_csv("census.csv")


display(data.head(n=2))

income_raw =data["income"]  #返回的是Series

features_raw = data.drop("income",axis = 1)


vs.distribution(features_raw)


skewed = ['capital-gain','capital-loss']

features_log_transformed = pd.DataFrame(data = features_raw)

pd.DataFrame(data=features_raw)
features_log_transformed[skewed]= features_raw[skewed].apply(lambda x: np.log(x+1))#lambda 匿名函数，x表示传入的参数

vs.distribution(features_log_transformed,transformed=True)
#归一化数字特征，放缩不会影响特征分布的形状
scaler  = MinMaxScaler() #default(0,1)
numerical =['age','education-num','capital-gain','capital-loss','hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

display(features_log_minmax_transform.head(n=5)) # 现在一些数字值都在0- 1之间了


final_features = pd.DataFrame(pd.get_dummies(features_log_minmax_transform))

display(final_features.head(n = 5))


#编写映射转换income_row
map_income={
    "<=50K":0,
    ">50K":1
}

binary_income = income_raw.map(map_income)
display(binary_income.head(n=5))




#exercise2
from sklearn.cross_validation import train_test_split


#Split hte f"features" and "income" data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(final_features,binary_income,test_size = 0.2,random_state = 0)

print("Traning set has %s samples" % format(X_train.shape[0]))
print("Testing set has %s samples" % format(X_test.shape[0]))



from sklearn import metrics

#Gaussion
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Gaussian Naive Bayes:")
print(metrics.classification_report(y_test,y_pred))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("LogisticRegression:")
print(metrics.classification_report(y_test,y_pred))
#Desicion Tree
from sklearn import tree
model  = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Dession Tree:")
print(metrics.classification_report(y_test,y_pred))

#KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("KNN:")
print(metrics.classification_report(y_test,y_pred))


#SVM
from sklearn import svm

model = svm.SVC()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("SVM:")
print(metrics.classification_report(y_test,y_pred))
