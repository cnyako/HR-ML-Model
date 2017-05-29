from pandas import read_csv
import pandas as pd
from sklearn.externals.joblib import load
import numpy

filename = 'HRTestFile.csv'
names = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'dept', 'salary']
dataset = read_csv(filename, names=names)
array = dataset.values
X = array[:,0:9]

# load the model from disk
loaded_model = load('finalized_model.sav')

df = pd.DataFrame(loaded_model.predict(X))

result = loaded_model.predict(X)

print(result)