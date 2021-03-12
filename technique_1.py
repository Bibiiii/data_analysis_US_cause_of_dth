import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

# Import processed data file
pre_processed_data = pd.read_csv (r'pre_processed.csv')

# Extract all columns from data file (not sure if necessary)

age_group = pd.DataFrame(pre_processed_data, columns= ['AgeGroup'])
#
# all_cause = pd.DataFrame(pre_processed_data, columns= ['AllCause'])
#
# analysis_date = pd.DataFrame(pre_processed_data, columns= ['AnalysisDate'])
#
# natural_cause = pd.DataFrame(pre_processed_data, columns= ['NaturalCause'])
#
sex = pd.DataFrame(pre_processed_data, columns= ['Sex'])

test = pre_processed_data.to_numpy()

test1 = age_group.to_numpy()

test2 = sex.to_numpy()

for i in range(0, len(test2)):
    if test2[i] == 'F':
        test2[i] = 0
    elif test2[i] == 'M':
        test2[i] = 1
    else:
        print("Error converting gender to binary number")

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
