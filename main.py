import numpy as np;
import pandas as pd;

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('mail_data.csv')
# print(df)

data = df.where(pd.notnull(df), '')
r = data.head(10)
print(r)