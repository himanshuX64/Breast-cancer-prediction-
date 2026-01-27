import streamlit as st
#import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
# visualization libraries

from sklearn.model_selection import train_test_split

# data modeling libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# data perprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score,f1_score,confusion_matrix,classification_report

st.title('Breast Cancer Prediction App')
#loading dataset
lbc=load_breast_cancer()

df=pd.DataFrame(lbc.data ,columns=lbc.features_name)
df

st.write('Hello world!')
