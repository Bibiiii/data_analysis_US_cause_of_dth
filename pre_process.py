from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')

'exec(%matplotlib inline)'
matplotlib.rcParams['figure.figsize'] = (12, 8)

pd.options.mode.chained_assignment = None

# read the data
df = pd.read_csv('input.csv')

# shape and data types of the data
print(df.shape)
print(df.dtypes)

#####################

# select numeric columns
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

# select non numeric columns
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)

#####################

##### check what columns are missing data #####

cols = df.columns
# specify the colours - yellow is missing. blue is not missing.
colours = ['#000099', '#ffff00']
print(df[cols].isnull())
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
plt.show()

# % of missing.
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

#####################

##### check for duplicate rows #####
print("Checking for duplicate rows...")
df_dedupped = df.drop_duplicates()
# check if there are duped rows (none)
print(df.shape)
print(df_dedupped.shape)
if (df.shape == df_dedupped.shape):
    print("No duplicates!")
else:
    print("Duplicates exist!")
#####################

# replace sex field with normalised value
sex_mapping = {
    "Male": 1,
    "Female": 2,
    "M": 1,
    "F": 2,
}

# replace race field (long names)
race_mapping = {
    "Hispanic": 1,
    "Non-Hispanic American Indian or Alaska Native": 2,
    "Non-Hispanic Asian": 3,
    "Non-Hispanic Black": 4,
    "Non-Hispanic White": 5,
    "Other": 6
}

# replace age field
age_mapping = {
    "0-4 years": 1,
    "5-14 years": 2,
    "15-24 years": 3,
    "25-34 years": 4,
    "35-44 years": 5,
    "45-54 years": 6,
    "55-64 years": 7,
    "65-74 years": 8,
    "75-84 years": 9,
    "85 years and over": 10
}

df = df.replace({"Sex": sex_mapping, "Race/Ethnicity": race_mapping,
                 "AgeGroup": age_mapping})


def fill_null_val(col):
    if col.name == "AllCause":
        return "UNK"  # this will be replaced by new totals
    elif "flag" in col.name:
        return 0
    else:
        return 4  # mean is 4.5


print("Replacing null values...")
df = df.apply(lambda col: col.fillna(fill_null_val(col)))  # fill empty cells

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

print("Replacing 'All Cause' column with new sum...")
df['AllCause'] = df[col_names].sum(axis=1).astype(
    int)  # replace 'All Cause' column with new totals

print("Dropping redundant columns...")
# drop the flag columns now they are used
df = df.drop(
    [col_name for col_name in df.columns if "flag" in col_name], axis=1)
df = df.drop("Jurisdiction of Occurrence", axis=1)

print("Saving new dataframe to csv...")
df.to_csv("pre_processed.csv")
print("Done!")

cols = df.columns
# specify the colours - yellow is missing. blue is not missing.
colours = ['#000099', '#ffff00']
sns.heatmap(df[cols].isnull(), vmin=0.0, vmax=1.0,
            cmap=sns.color_palette(colours))
plt.show()

# SAVE TRAINING AND TEST SETS
df_shuffled = df.sample(frac=1, random_state=1)  # shuffle dataset
train, test = train_test_split(df_shuffled, test_size=0.2) # split data 80:20 (train:test)
train.reset_index(drop=True).to_csv("train.csv", index=False)
test.reset_index(drop=True).to_csv("test.csv", index=False)
