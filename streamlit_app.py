import streamlit as st
#import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
# visualization libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# data modeling libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# data perprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score,f1_score,confusion_matrix,classification_report

st.title('Breast Cancer Prediction App')
#loading dataset
df=pd.read_csv("https://raw.githubusercontent.com/himanshuX64/CampusX/refs/heads/main/dataset/breast-cancer.csv")
X=df.drop('diagnosis',axis=1)
y=df['diagnosis']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
encoder=LabelEncoder()
X_train=encoder.fit_transform(X_train)
X_test=encoder.transform(X_test)

with st.expander('Data'):
  df

with st.expander("Depandent Data ")
  X

with st.expander("Target Data")
  y

