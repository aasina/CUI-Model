from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
import pandas as pd

st.title('CUI Machine Learning V0.1')

st.write("""
#This is trial of Machine Learning Implementation for Determination of CUI Probability of Failure (PoF)

There are several main parameter in determining CUI PoF in this model. The parameters are temperature (°C), diameter (mm), cyclic temperature operation and insulation type.

""")

st.sidebar.header('Masukkan parameter input')

cuidata = pd.read_csv('datamodel110121.csv')

# define feature column or parameter that being utilized for prediction
feature_col = ['tempc', 'dia', 'cuicr', 'cyclic', 'insul']
X = cuidata[feature_col]

# define target column
y = cuidata['pof']


def user_input_features():
    tempc = st.sidebar.slider('Temperatur (°C)', -150, 150, 35)
    dia = st.sidebar.slider('Diameter (mm)', 0, 350, 50)
    cuicr = st.sidebar.slider('Actual CUI rate (mm/yr)', 0.0, 1.0, 0.01)
    cyclic = st.sidebar.slider('Cyclic Operation', 0, 1, 0)
    insul = st.sidebar.slider('Type of Insulation', 0, 2, 1)
    data = {'Temperatur (°C)': tempc,
            'Diameter (mm)': dia,
            'Actual CUI rate (mm/yr)': cuicr,
            'Cyclic Operation': cyclic,
            'Type of Insulation': insul}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

cuidata = pd.read_csv('datamodel110121.csv')

# define feature column or parameter that being utilized for prediction
feature_col = ['tempc', 'dia', 'cuicr', 'cyclic', 'insul']
X = cuidata[feature_col]

# define target column
y = cuidata['pof']

# STEP 1: split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=4)


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

prediction = knn.predict(df)
prediction_proba = knn.predict_proba(df)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

# plot the relationship between K and testing accuracy

fig, ax = plt.subplots()
ax.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

st.pyplot(fig)


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
y_pred2 = knn.predict(X_test)
knnaccuracy = metrics.accuracy_score(y_test, y_pred2)  # change to streamlit

st.write("""
Nilai Akurasi Maksimum Model menggunakan KNN adalah """
         )

knnaccuracy


# STEP 2: train the model on the training set
logreg = LogisticRegression(solver='newton-cg', C=30.0, random_state=0)
logreg.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred3 = logreg.predict(X_test)

# compare actual response values (y_test) with predicted response values (y_pred)
#print(metrics.accuracy_score(y_test, y_pred3))

# try K=1 through K=25 and record testing accuracy
c_range = list(range(1, 31))
scores = []
for c in c_range:
    logreg = LogisticRegression(solver='newton-cg', C=c, random_state=0)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

# plot the relationship between K and testing accuracy
plt.plot(c_range, scores)
plt.xlabel('Value of Regularization for Logistic Regression')
plt.ylabel('Testing Accuracy')

#print(classification_report(y_test, logreg.predict(X_test)))

#print(classification_report(y_test, knn.predict(X_test)))
