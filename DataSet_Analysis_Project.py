import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
st.title("Machine Learning Analysis ")
st.subheader("Uploading DataSet")
data = st.text_input("File path / name : ")
st.subheader("Insights of DataSet")
df = pd.read_csv(data)
st.write("The DataSet is : ")
st.table(df.head())
st.write("The shape of the DataSet is :")
st.write(df.shape)
st.write("The correlation between attributes is :")
st.table(df.corr())
st.write("The description about the data is ")
st.write(df.describe())
st.header("Regression : ")
output = st.text_input("output_columns is ")
sizes = st.number_input("select split sizes are ")
st.subheader("Regression Algorithms")
if st.button("Linear Regression"):
	data = pd.read_csv(data)
	x = data.drop(output, axis=1).values
	y = data[output].values
	X=data.select_dtypes(include=np.number)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizes, random_state=42)
	reg = linear_model.LinearRegression()
	reg.fit(X_train, y_train)
	y_pred1=reg.predict(X_train)
	ans = r2_score(y_train, y_pred1)
	st.write(ans)
st.header("Classification : ")
output_label = st.text_input("output_column")
size = st.number_input("select split size")
n = st.number_input("number of neighbours for knn", min_value=1, max_value=101, value=5, step=2)
et = st.number_input("number of estimators",min_value=1, max_value=101, value=5, step=1)
st.subheader("Classification Algorithms")
if st.button("KNN"):
	data = pd.read_csv(data)
	X = data.drop(output_label, axis=1).values
	y = data[output_label].values
	X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=size, random_state=41)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	knn = KNeighborsClassifier(n)
	knn.fit(X_train, y_train)
	X_test = scaler.transform(X_test)
	y_pred = knn.predict(X_test)
	ans = knn.score(X_test, y_test)
	st.write(ans)
if st.button("SVM"):
	data = pd.read_csv(data)
	X = data.drop(output_label, axis=1).values
	y = data[output_label].values
	X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=size, random_state=41)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	model = SVC()
	model.fit(X_train, y_train)
	X_test = scaler.transform(X_test)
	y_pred = model.predict(X_test)
	ans = model.score(X_test, y_test)
	st.write(ans)
if st.button("DecisionTree"):
	data = pd.read_csv(data)
	X = data.drop(output_label, axis=1).values
	y = data[output_label].values
	X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=size, random_state=0)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	model = DecisionTreeClassifier()
	model.fit(X_train, y_train)
	X_test = scaler.transform(X_test)
	y_pred = model.predict(X_test)
	ans = model.score(X_test, y_test)
	st.write(ans)
if st.button("Logistic Regression"):
	data = pd.read_csv(data)
	X = data.drop(output_label, axis=1).values
	y = data[output_label].values
	X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=size, random_state=0)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	model = LogisticRegression(C=0.09,random_state = 41)
	model.fit(X_train, y_train)
	X_test = scaler.transform(X_test)
	y_pred = model.predict(X_test)
	ans = model.score(X_test, y_test)
	st.write(ans)
if st.button("Naive Bayes"):
	data = pd.read_csv(data)
	X = data.drop(output_label, axis=1).values
	y = data[output_label].values
	X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=size, random_state=0)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	model = GaussianNB()
	model.fit(X_train, y_train)
	X_test = scaler.transform(X_test)
	y_pred = model.predict(X_test)
	ans = model.score(X_test, y_test)
	st.write(ans)
if st.button("Random Forest"):
	data = pd.read_csv(data)
	X = data.drop(output_label, axis=1).values
	y = data[output_label].values
	X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=size, random_state=0)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	model = RandomForestClassifier(et,criterion="entropy")
	model.fit(X_train, y_train)
	X_test = scaler.transform(X_test)
	y_pred = model.predict(X_test)
	ans = model.score(X_test, y_test)
	st.write(ans)
st.header('Clustering')
if st.button('K-Means'):
	data = pd.read_csv(data)