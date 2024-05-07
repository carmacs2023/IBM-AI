from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import normalize, StandardScaler
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import ml
import snapml
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import pprint
import time
import warnings
warnings.filterwarnings('ignore')


def download(url, filename):
    # Send a GET request to the specified URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and write the content of the response
        with open(filename, 'wb') as f:
            f.write(response.content)

# region regression

# path = ("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/"
#         "labs/Module%202/data/FuelConsumptionCo2.csv")
# file = "FuelConsumption.csv"
# download(path, file)

# # Read data
# df = pd.read_csv("FuelConsumption.csv")
#
# # summarize the data
# df.head()
# print(df.describe())
#
# # create a new dataframe with selected features
# cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# print(cdf.head(9))  # print first 9 rows of the data

# # plot each of these features
# viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
# viz.hist()
# plt.show()
#
# # plot each of these features vs the Emission, to see how linear is their relation
#
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()
#
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

# Example usage simple_linear_regression
# Assuming 'df' is your DataFrame, and it contains 'ENGINESIZE' or 'FUELCONSUMPTION_COMB' and 'CO2EMISSIONS' columns
# ml.simple_linear_regression(df, 0.8, 'ENGINESIZE', 'CO2EMISSIONS')
# ml.simple_linear_regression(df, 0.6, 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS')

# Example usage multiple_linear_regression
# df is your DataFrame containing 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', and 'CO2EMISSIONS'
# ml.linear_regression(df, 0.8, ['ENGINESIZE'], 'CO2EMISSIONS')  # Simple linear regression
# ml.linear_regression(df, 0.8, ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB'], 'CO2EMISSIONS')  # Multiple linear regression

# endregion

# region KNN

# file = "telecom_customer_dataset.csv"
# # path = ("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/"
# #         "labs/Module%203/data/teleCust1000t.csv")
# # download(path, file)
# df = pd.read_csv(file)
#
# # ml.knn(df, 'custcat', k=8, test_size=0.2, random_state=4)
# ml.knn(df, 'custcat', k=8, test_size=0.2, random_state=4, optimize_k=50)

# endregion

# region decission trees Part I - multi-class classifiers


# file = "drug200.csv"
# # path = ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs'
# #         '/Module%203/data/drug200.csv')
# # # download(path, file)
# df = pd.read_csv(file, delimiter=",")
#
# label_encoder_dict = {
#     'Sex': preprocessing.LabelEncoder(),
#     'BP': preprocessing.LabelEncoder(),
#     'Cholesterol': preprocessing.LabelEncoder()
# }
#
# ml.decision_tree_classifier(df=df, target_col_name='Drug', test_size=0.3, random_state=3, criterion='entropy', max_depth=4,
#                             label_encoder_dict=label_encoder_dict,
#                             export_graph=True, target_class_distribution='balanced',
#                             replicas=10, normalize=False, standardize=False,
#                             library='sklearn', eval_metrics=True)

# endregion


# region decission trees part II - binary classes, sklearmn vs snapml

file = "creditcard.csv"
df = pd.read_csv(file)

# sklearn
# ml.decision_tree_classifier(df=df, target_col_name='Class', test_size=0.3, random_state=35, criterion='gini', max_depth=4,
#                             label_encoder_dict=None,
#                             export_graph=True, target_class_distribution='unbalanced',
#                             drop_col=['Time'], replicas=10, normalize=True, standardize=True,
#                             library='sklearn', eval_metrics=True)

# snapml (about 12x faster even if using CPU only than sklearn)
ml.decision_tree_classifier(df=df, target_col_name='Class', test_size=0.3, random_state=35, criterion='gini', max_depth=4,
                            label_encoder_dict=None,
                            export_graph=False, target_class_distribution='unbalanced',
                            drop_col=['Time'], replicas=10, normalize=True, standardize=True,
                            library='snapml', eval_metrics=True)
# endregion



# # import the linear Support Vector Machine (SVM) model from Scikit-Learn
# from sklearn.svm import LinearSVC
#
# # instatiate a scikit-learn SVM model
# # to indicate the class imbalance at fit time, set class_weight='balanced'
# # for reproducible output across multiple function calls, set random_state to a given integer value
# sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
#
# # train a linear Support Vector Machine model using Scikit-Learn
# t0 = time.time()
# sklearn_svm.fit(X_train, y_train)
# sklearn_time = time.time() - t0
# print("[Scikit-Learn] Training time (s):  {0:.2f}".format(sklearn_time))
#
# # import the Support Vector Machine model (SVM) from Snap ML
# from snapml import SupportVectorMachine
#
# # in contrast to scikit-learn's LinearSVC, Snap ML offers multithreaded CPU/GPU training of SVMs
# # to use the GPU, set the use_gpu parameter to True
# # snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, use_gpu=True, fit_intercept=False)
#
# # to set the number of threads used at training time, one needs to set the n_jobs parameter
# snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
# # print(snapml_svm.get_params())
#
# # train an SVM model using Snap ML
# t0 = time.time()
# model = snapml_svm.fit(X_train, y_train)
# snapml_time = time.time() - t0
# print("[Snap ML] Training time (s):  {0:.2f}".format(snapml_time))
#
# # compute the Snap ML vs Scikit-Learn training speedup
# training_speedup = sklearn_time/snapml_time
# print('[Support Vector Machine] Snap ML vs. Scikit-Learn training speedup : {0:.2f}x '.format(training_speedup))
#
# # run inference using the Scikit-Learn model
# # get the confidence scores for the test samples
# sklearn_pred = sklearn_svm.decision_function(X_test)
#
# # evaluate accuracy on test set
# acc_sklearn = roc_auc_score(y_test, sklearn_pred)
# print("[Scikit-Learn] ROC-AUC score:   {0:.3f}".format(acc_sklearn))
#
# # run inference using the Snap ML model
# # get the confidence scores for the test samples
# snapml_pred = snapml_svm.decision_function(X_test)
#
# # evaluate accuracy on test set
# acc_snapml = roc_auc_score(y_test, snapml_pred)
# print("[Snap ML] ROC-AUC score:   {0:.3f}".format(acc_snapml))


