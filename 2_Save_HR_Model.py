# Save Model Using joblib
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load

filename = 'HR.csv'
names = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'dept', 'salary', 'left']
dataset = read_csv(filename, names=names)
array = dataset.values
X = array[:,0:9]
Y = array[:,9]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# Fit the model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# save the model to disk
filename = 'finalized_model.sav'
dump(model, filename)


