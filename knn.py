from collections import Counter
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# to report on precision and recall
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

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
enum_cols = {name: i for i, name in enumerate(col_names)}


def get_train_test(data):
    # split data 80:20 (train:test)
    train, test = train_test_split(data, test_size=0.2)
    return (train, test)


def get_x_y(df):
    """
    Takes dataframe and returns relevant features and labels
    """
    feature_names = [
        "Sex",
        "Race/Ethnicity",
        "AgeGroup",
        "Date Of Death Month"  # reduces accuracy, f scores, increases weighted avg
    ]

    x = df[feature_names].to_numpy()
    # get column name with highest death rate
    relevant_cols = [key for key in enum_cols if key != "NaturalCause"]
    y = df[relevant_cols].idxmax(axis=1).replace(
        enum_cols).to_numpy()  # use enum mapping
    y.to_csv("knn_cod.csv", index=False)
    return (x, y)


def train_classifier(features, labels):
    print("Training Classifier...")
    n_neighbors = 10
    # no difference between distance and uniform except when using month data
    clf = KNeighborsClassifier(n_neighbors, weights='uniform')
    return clf.fit(features, labels)


def predict_labels(samples, classifier):
    print("Predicting Labels...")
    return classifier.predict(samples)


def confusion_matrix_heatmap(y_test, preds, classification_labels):
    """Function to plot a confusion matrix"""
    labels = list(set(y_test))
    long_labels = [ll + " (" + str(l) + ")" for ll, l
                   in zip(classification_labels, labels)]
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(long_labels)

    for i in range(len(cm)):
        for j in range(len(cm)):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    # fig.tight_layout()
    plt.show()


def get_classifier_report(predictions, groundtruth):
    labels = list(
        map(lambda cat: [k for k in enum_cols if enum_cols[k] == cat][0], list(set(groundtruth))))  # map numbers to string labels

    print(classification_report(groundtruth, predictions,
                                target_names=labels, labels=list(set(groundtruth))))
    confusion_matrix_heatmap(groundtruth, predictions, labels)


def cross_validate(training_data):
    print("Cross-validating classifier...")
    train_data, test_data = get_train_test(training_data)
    train_x, train_y = get_x_y(training_data)
    test_x, test_y = get_x_y(training_data)

    print(train_x.shape, train_y.shape)  # ensure are same
    # train on training samples
    classifier = train_classifier(train_x, train_y)
    predictions = predict_labels(test_x, classifier)  # predict on test samples

    get_classifier_report(predictions, test_y)


_TESTING_ = False


def main():
    # Import processed data file
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    cross_validate(train_df)

    # run trained classifier on test set
    if _TESTING_ == True:
        print("TESTING CLASSIFIER ON TEST SET...")
        train_x, train_y = get_x_y(train_df)
        test_x, test_y = get_x_y(test_df)

        classifier = train_classifier(train_x, train_y)
        predictions = predict_labels(test_x, classifier)

        get_classifier_report(predictions, test_y)


if __name__ == "__main__":
    main()
