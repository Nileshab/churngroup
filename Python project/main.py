import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.columns = df.columns.str.lower()
df['totalcharges'] = df['totalcharges'].replace(' ', 0)
df['seniorcitizen'] = df['seniorcitizen'].astype('object')
df['totalcharges'] = df['totalcharges'].astype('float')
df.drop(columns=['customerid'], inplace=True)
df.drop_duplicates(inplace=True)
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
df['tenure_group'] = pd.cut(df.tenure, bins=3, labels=['low', 'medium', 'high'])
df['monthlycharges_group'] = pd.cut(df.monthlycharges, bins=3, labels=['low', 'medium', 'high'])
df['totalcharges_group'] = pd.cut(df.totalcharges, bins=3, labels=['low', 'medium', 'high'])
# print(df.head())
X = df.drop(['churn', 'tenure', 'monthlycharges', 'totalcharges'], axis=1)
y = df['churn']
print(X.columns)
l = LabelEncoder()
for i in X.columns:
  df[i] = l.fit_transform(df[i])
X = df.drop(['churn', 'tenure', 'monthlycharges', 'totalcharges'], axis=1)
y = df['churn']
print(df.head())
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)

st = StandardScaler()
x_train_0 = st.fit_transform(x_train)
x_test_0 = st.transform(x_test)
smote = SMOTE(sampling_strategy=0.8,k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(x_train_0, y_train)

base_estimator = DecisionTreeClassifier(class_weight='balanced', max_depth=1)
adb = AdaBoostClassifier(base_estimator,n_estimators=100)
adb.fit(X_train_res, y_train_res)
print(f"ADABOOST MODEL : {classification_report(y_test,adb.predict(x_test_0))})")

from sklearn.svm import SVC

svm = SVC(kernel='poly', C=100)
svm.fit(X_train_res, y_train_res)

print('Train score:', svm.score(X_train_res, y_train_res))
print('Test score:', svm.score(x_test_0, y_test))
print("SVM MODEL : f{classification_report(y_test,adb.predict(x_test_0)))")




# ********************************************************************* (Above Code is Done for Converting into categroies)
newdf = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
newdf.drop(columns=['customerID'], inplace=True)
newdf.columns = newdf.columns.str.lower()
newdf['totalcharges'] = newdf['totalcharges'].replace(' ', 0)
newdf['seniorcitizen'] = newdf['seniorcitizen'].astype('object')
newdf['totalcharges'] = newdf['totalcharges'].astype('float')
newdf.drop_duplicates(inplace=True)
num_cols = newdf.select_dtypes(include='number').columns.tolist()
cat_cols = newdf.select_dtypes(include='object').columns.tolist()

l = LabelEncoder()
for i in cat_cols:
  newdf[i] = l.fit_transform(newdf[i])
X = newdf.drop(['churn'], axis=1)
y = newdf['churn']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.22)
st = StandardScaler()
x_train_0 = st.fit_transform(x_train)
x_test_0 = st.transform(x_test)
smote = SMOTE(sampling_strategy=0.8,k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(x_train_0, y_train)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.01)
logreg.fit(x_train,y_train)

# print('Train score:', logreg.score(x_train,y_train))
# print('Test score:', logreg.score(x_test, y_test))
print(classification_report(y_test,logreg.predict(x_test)))


svm = SVC(kernel='poly', C=100)
svm.fit(x_train, y_train)
print(f"SVM MODEL : {classification_report(y_test,adb.predict(x_test))})")

base_estimator = DecisionTreeClassifier(class_weight='balanced', max_depth=1)
adb = AdaBoostClassifier(base_estimator,n_estimators=100)
adb.fit(x_train, y_train)
print(f"ADABOOST MODEL : {classification_report(y_test,adb.predict(x_test))})")