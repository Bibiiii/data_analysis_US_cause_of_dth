import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

col_names = [
    "NaturalCause",
    "Septicemia (A40-A41)",
    "Malignant neoplasms (C00-C97)",
    "Diabetes mellitus (E10-E14)",
    "Alzheimer disease (G30)",
    "Influenza and pneumonia (J09-J18)",
    "Chronic lower respiratory diseases (J40-J47)",
    "Other diseases of respiratory system (J00-J06,J30-J39,J67,J70-J98)",
    "Nephritis, nephrotic syndrome and nephrosis (N00-N07,N17-N19,N25-N27)",
    "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (R00-R99)",
    "Diseases of heart (I00-I09,I11,I13,I20-I51)",
    "Cerebrovascular diseases (I60-I69)",
    "COVID-19 (U071, Multiple Cause of Death)",
    "COVID-19 (U071, Underlying Cause of Death)"
]

def get_train_test(df):
    """
    Takes dataframe and returns train and test data from the dataframe
    """
    train = df[["Sex", "Race/Ethnicity", "AgeGroup"]
               ].to_numpy()
    test = df[col_names].to_numpy()
    return (train, test)

# Import processed data file
pre_processed_data = pd.read_csv(r'pre_processed.csv')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_x, train_y = get_train_test(train_df)
test_x, test_y = get_train_test(test_df)

# Extract all columns from data file (not sure if necessary)

age_group = pd.DataFrame(pre_processed_data, columns=['AgeGroup'])
#
# all_cause = pd.DataFrame(pre_processed_data, columns= ['AllCause'])
#
# analysis_date = pd.DataFrame(pre_processed_data, columns= ['AnalysisDate'])
#
# natural_cause = pd.DataFrame(pre_processed_data, columns= ['NaturalCause'])
#
sex = pd.DataFrame(pre_processed_data, columns=['Sex'])

test = pre_processed_data.to_numpy()

test1 = age_group.to_numpy()

test2 = sex.to_numpy()

X = np.zeros((len(test1), 2))

for i in range(0, len(test2)):
    X[i][0] = test1[i]
    X[i][1] = test2[i]


nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

print(indices)

print(distances)

plt.plot(indices, distances)

plt.show()
