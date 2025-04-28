import numpy as np;
import pandas as pd;

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('mail_data.csv')
# print(df)

data = df.where(pd.notnull(df), '')
# r = data.head(10)
# print(r) return first 10 data sets

# print(data.info()) #meta data about the data


data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

X, Y = data['Message'], data['Category']

# spliting the data into test and train dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=3)


# Vectorization
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train =Y_train.astype('int')
Y_test = Y_test.astype('int')

# traing a model using logistic regression
model = LogisticRegression()
model.fit(X_train_features, Y_train)

prediction_on_traing_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_traing_data)

# print("Accuracy on training data : ", accuracy_on_training_data) #0.97

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

# print("Accuracy on test data : ", accuracy_on_test_data) 


input_your_mail = ["Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize now: http://bit.ly/spamlink"]
input_your_mail2 =["Hi John, just wanted to check if we're still meeting for lunch tomorrow. Let me know what time works best for you."]



input_data_features = feature_extraction.transform(input_your_mail)
input_data_features2 = feature_extraction.transform(input_your_mail2)

prediction = model.predict(input_data_features)
prediction2 = model.predict(input_data_features2)
print("Prediction 1:", "spam" if prediction == 0 else "Not spam")
print("Prediction 2:", "spam" if prediction2 == 0 else "Not spam")
