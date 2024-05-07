import pandas as pd
import numpy as np
# import pylab as pl
# import requests
# import sys
import os
import subprocess
# import pprint
import time
import matplotlib.pyplot as plt
# sklearn
from sklearn import linear_model, preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn.svm import LinearSVC
# snapml
from snapml import DecisionTreeClassifier as snapml_DecisionTreeClassifier


def simple_linear_regression(df: pd.DataFrame, mask_percent: float, col_x: str, col_y: str):
    """
    Performs simple linear regression on a given dataset with a split of train and test data, plotting the results.

    Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' between
    the actual value y in the dataset, and the predicted value yhat using linear approximation.

    Parameters:
    df (pd.DataFrame): The dataset containing the relevant columns.
    mask_percent (float): The percentage of the data to include in the training set (0 to 1).
    col_x (str): The column name in the dataframe to use as the independent variable.
    col_y (str): The column name in the dataframe to use as the dependent variable.

    Returns:
    None
    """
    # Create train and test SPLIT datasets
    msk = np.random.rand(len(df)) < mask_percent
    train = df[msk]
    test = df[~msk]

    # Plot train data distribution
    plt.scatter(train[col_x], train[col_y], color='blue')
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title("Train data distribution")
    plt.show()

    # Modeling
    regression_model = linear_model.LinearRegression()
    train_x = np.asanyarray(train[[col_x]])
    train_y = np.asanyarray(train[[col_y]])
    regression_model.fit(train_x, train_y)

    # Coefficients and intercept
    print('Coefficients: ', regression_model.coef_)
    print('Intercept: ', regression_model.intercept_)

    # Plot outputs
    plt.scatter(train[col_x], train[col_y], color='blue')
    plt.plot(train_x, regression_model.coef_[0][0]*train_x + regression_model.intercept_[0], '-r')
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title("Regression Line")
    plt.show()

    # Evaluation

    # We compare the actual values and predicted values to calculate the accuracy of a regression model.
    # Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.
    # There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:
    #   Mean Absolute Error:        Mean of the abs value of the errors. Easiest metric to understand since it’s just average error.
    #   Mean Squared Error (MSE):   Mean Squared Error (MSE) is the mean of the squared error.
    #                               more popular because the focus is geared more towards large errors.
    #                               This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
    #   Root Mean Squared Error (RMSE).
    #   R-squared:                  Although not an error, it is popular metric to measure the performance of your regression model.
    #                               It represents how close the data points are to the fitted regression line.
    #                               The higher the R-squared value, the better the model fits your data.
    #                               The best possible score is 1.0, and it can be negative (because the model can be arbitrarily worse).

    # Prediction - Test the model
    test_x = np.asanyarray(test[[col_x]])
    test_y_ = regression_model.predict(test_x)  # Predict the values
    test_y = np.asanyarray(test[[col_y]])

    # Print evaluation metrics
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y, test_y_))


def linear_regression(df: pd.DataFrame, mask_percent: float, col_x: list, col_y: str):
    """
    Performs linear regression (simple or multiple) on a given dataset, plotting the results and printing evaluation metrics.

    Parameters:
    df (pd.DataFrame): The dataset containing the relevant columns.
    mask_percent (float): The percentage of the data to include in the training set (0 to 1).
    col_x (list): The column name or list of column names in the dataframe to use as the independent variables.
    col_y (str): The column name in the dataframe to use as the dependent variable.

    Returns:
    None
    """
    # Create train and test datasets
    msk = np.random.rand(len(df)) < mask_percent
    train = df[msk]
    test = df[~msk]

    # Modeling
    regression_model = linear_model.LinearRegression()
    train_x = np.asanyarray(train[col_x])
    train_y = np.asanyarray(train[[col_y]])
    regression_model.fit(train_x, train_y)

    # Coefficients and intercept
    print('Coefficients: ', regression_model.coef_)
    print('Intercept: ', regression_model.intercept_)

    # Plot outputs for simple linear regression only
    if len(col_x) == 1:
        plt.scatter(train[col_x], train[col_y], color='blue')
        plt.plot(train_x, regression_model.coef_[0] * train_x + regression_model.intercept_, '-r')
        plt.xlabel(col_x[0])
        plt.ylabel(col_y)
        plt.title("Regression Line")
        plt.show()

    # Evaluation
    test_x = np.asanyarray(test[col_x])
    test_y = np.asanyarray(test[[col_y]])
    test_y_ = regression_model.predict(test_x)

    # Print evaluation metrics
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y, test_y_))

    # Additional metrics for multiple regression
    if len(col_x) > 1:
        print('Variance score: %.2f' % regression_model.score(test_x, test_y))


def knn(df: pd.DataFrame, target_col_name: str, k: int, test_size: float, random_state: int, optimize_k: int = None):

    """
    Apply K-Nearest Neighbors classification to the provided DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset to classify.
    target_col_name (str): The name of the target variable column.
    k (int): The number of nearest neighbors to use for classification.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator for reproducibility.
    optimize_k (int, optional): If provided, optimizes k from 1 to this value, otherwise uses the provided k.

    Returns:
    None: Outputs the classification results directly.
    """
    # DataFrame columns
    print(f'dataframe columns {df.columns.tolist()}')

    # Class distribution
    print(f'value count by target column\n {df[target_col_name].value_counts()}')

    # Converting DataFrame to Numpy array for features and labels
    x = df[df.columns.tolist()].values  # Dataset
    # x = df.drop(columns=[target_col_name]).values  # Remove the target column for feature set
    y = df[target_col_name].values      # Target variable

    # Normalize the feature set
    x = preprocessing.StandardScaler().fit_transform(x.astype(float))

    # Splitting the dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    print('Train set:', x_train.shape, y_train.shape)
    print('Test set:', x_test.shape, y_test.shape)

    # K optimization
    if optimize_k:
        accuracies = []
        ks = range(1, optimize_k + 1)
        for k in ks:
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(x_train, y_train)
            yhat = neigh.predict(x_test)
            accuracy = metrics.accuracy_score(y_test, yhat)
            accuracies.append(accuracy)
            print(f"k={k}: Train set Accuracy: {metrics.accuracy_score(y_train, neigh.predict(x_train))}")
            print(f"k={k}: Test set Accuracy: {accuracy}")

        # Plotting the accuracy for different k values
        plt.figure(figsize=(10, 6))
        plt.plot(ks, accuracies, marker='o', linestyle='-', color='b')
        plt.title('Accuracy vs. Number of Neighbors (k)')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Test Set Accuracy')
        plt.grid(True)
        plt.show()

    else:
        # K-Nearest Neighbors Modeling
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train, y_train)  # Training the model

        # Predicting the test set results
        yhat = neigh.predict(x_test)
        print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train)))
        print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


def decision_tree_classifier(df: pd.DataFrame, target_col_name: str, test_size: float, random_state: int, criterion: str = "entropy",
                             max_depth: int = None, dot_filename: str = "tree.dot", label_encoder_dict: dict = None, export_graph: bool = True,
                             target_class_distribution: str = 'balanced', drop_col: list = None, normalize: bool = False, standardize: bool = False,
                             replicas: int = None, library: str = 'sklearn', eval_metrics: bool = True):
    """
    Train and visualize a decision tree classifier.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col_name (str): The name of the target variable column to classify.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator for reproducibility.
        criterion (str): The function to measure the quality of a split.
            Default is "entropy" for information gain, slow due to logarithmic calculations,
            alternative is "gini" that measure the impurity of the node.
        max_depth (int): The maximum depth of the tree.
        dot_filename (str): The filename for the exported dot file.
        label_encoder_dict (dict): Dictionary of column names and their respective LabelEncoder() instances.
            Sklearn Decision Trees does not handle categorical variables.
            Use LabelEncoder() method to convert these features to numerical values.
            for example Low High to 0 and 1.
        export_graph (bool): Enable or Disables exportation of decision tree graph
        target_class_distribution (str): balanced or unbalanced.
            if unbalanced ensures random split of the data with same proportion of each class as observed in the original dataset.
        drop_col (list) : list of columns to drop from the dataframe
        normalize (bool) : apply data normalization
        standardize (bool) : standardize features by removing the mean and scaling to unit variance
        replicas (int): inflates dataset for testing with replicas
        library (str): 'sklearn' or 'snapml'
        eval_metrics (bool): evaluate model for accuracy and ROC-AUC metrics

    Example:
    file = "drug200.csv"
    df = pd.read_csv(file, delimiter=",")

    if df has the following columns ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug'] where 'Drug' is the target column

    label_encoder_dict = {
        'Sex': preprocessing.LabelEncoder(),
        'BP': preprocessing.LabelEncoder(),
        'Cholesterol': preprocessing.LabelEncoder()
    }
    ml.decision_tree_classifier(df, 'Drug', test_size=0.3, random_state=3, max_depth=4, label_encoder_dict=label_encoder_dict)
    """

    if drop_col:
        df = drop_columns(df=df, drop_col=drop_col)

    if replicas:
        df = pd.DataFrame(np.repeat(df.values, replicas, axis=0), columns=df.columns)

    # Dataset description"
    print(f'dataset description: \n{df.head()}')
    print(f'shape: {df.shape}')
    print(f'observations: {str(len(df))}')
    print(f'variables {str(len(df.columns))}')
    labels = df[target_col_name].unique()
    sizes = df[target_col_name].value_counts().values
    print(f'labels : {labels}')  # get the set of distinct target column values
    print(f'sizes : {sizes}')  # get the count for each of the target column values

    # plot Target column value counts
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.3f%%')
    ax.set_title('Target Column Value Counts')
    plt.show()

    # Encode categorical variables using LabelEncoder
    if label_encoder_dict:
        print(f'Before using label encoder:\n {df.drop(labels=target_col_name, axis=1).values[0:5]}')
        for column, encoder in label_encoder_dict.items():
            df[column] = encoder.fit_transform(df[column])
        print(f'After using label encoder:\n {df.drop(labels=target_col_name, axis=1).values[0:5]}')

    # Splitting the dataset into features X and target variable Y
    x = df.drop(labels=target_col_name, axis=1).values
    y = df[target_col_name].values

    # standardize features by removing the mean and scaling to unit variance
    if standardize:
        scaler = preprocessing.StandardScaler()
        x = scaler.fit_transform(X=x)

    # data normalization
    if normalize:
        x = preprocessing.normalize(x, norm="l1")

    # Splitting the data into train and test sets
    if target_class_distribution == 'balanced':
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        w_train = None
    elif target_class_distribution == 'unbalanced':
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
        # compute the sample weights to be used as input to the train routine to take into account the class imbalance of the dataset
        w_train = compute_sample_weight(class_weight='balanced', y=y_train)

    print('x_train.shape=', x_train.shape, 'y_train.shape=', y_train.shape)
    print('x_test.shape=', x_test.shape, 'y_test.shape=', y_test.shape)

    # Modeling the Decision Tree
    if library == 'sklearn':
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
    elif library == 'snapml':
        # CPU training
        # to set the number of CPU threads used at training time, set the n_jobs parameter
        tree = snapml_DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)

        # GPU training - only supports Nvidia cards with CUDA support
        # Snap ML offers multi-threaded CPU/GPU training of decision trees, unlike scikit-learn
        # tree = snapml_DecisionTreeClassifier(max_depth=4, random_state=45, use_gpu=True)
    else:
        print(f'Error: library does not exist, please select "sklearn" or "snapml".')
        return None

    # Training the model
    t0 = time.time()
    # Note: typehints below for x and y are not used to be able to make it work for both snapml or sklearn inter-changeably
    tree.fit(x_train, y_train, sample_weight=w_train)
    learn_time = time.time() - t0
    print(f'{library}'+" Training time (s):  {0:.5f}".format(learn_time))  # Estimate computation time

    # Predicting the test set results
    y_pred = tree.predict(X=x_test)

    # Evaluating the model
    if eval_metrics:
        print("Decision Tree Accuracy score: ", accuracy_score(y_true=y_test, y_pred=y_pred))

        num_classes = len(labels)
        if num_classes == 2:
            # ROC - AUC or (Receiver Operating Characteristic - Area Under Curve)
            # To calculate ROC-AUC score predicted probabilities are needed instead of hard class labels.
            # The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.
            # ROC-AUC evaluates a classifier's performance across all possible classification thresholds.

            # run inference and compute the probabilities of the test samples
            # to belong to the class of fraudulent transactions
            y_pred_prob = tree.predict_proba(X=x_test)[:, 1]

            # evaluate the Compute Area Under the Receiver Operating Characteristic
            # Curve (ROC-AUC) score from the predictions
            # roc_auc = roc_auc_score(y_test, y_pred_prob)
            # print(f'{library}' + "ROC-AUC score : {0:.3f}".format(roc_auc))

            print("Decision Tree ROC-AUC Score: ", roc_auc_score(y_true=y_test, y_score=y_pred_prob))
        else:
            print('ROC-AUC only supported for binary class')

    # Export the trained model to a Graphviz format
    if export_graph:

        if library == 'sklearn':
            export_graphviz(decision_tree=tree, out_file=dot_filename, filled=True, feature_names=df.drop(target_col_name, axis=1).columns.tolist())
            generate_tree_image(dot_file=dot_filename, output_image=dot_filename.replace('.dot', '.png'))

            # Open the tree image; adjust the command according to your OS
            os.system(f'open {dot_filename.replace(".dot", ".png")}')
        else:
            print('exporting the model to a Graphviz format not possible with snapml library')


def drop_columns(df: pd.DataFrame, drop_col: list = None) -> pd.DataFrame:
    """
    Creates a new DataFrame by dropping specified columns.

    Parameters:
    df (pd.DataFrame): The original DataFrame.
    drop_col (list, optional): A list of column names to be dropped. Default is None.

    Returns:
    pd.DataFrame: A new DataFrame with specified columns removed.
    :rtype: pd.DataFrame
    """
    if drop_col and all(col in df.columns for col in drop_col):
        # Drop the specified columns if all are present in the DataFrame
        return df.drop(columns=drop_col)
    elif drop_col:
        # Handle the case where some columns might not exist in the DataFrame
        valid_columns = [col for col in drop_col if col in df.columns]
        return df.drop(columns=valid_columns)
    else:
        # If drop_col is None or empty, return the original DataFrame unchanged
        return df.copy()


def generate_tree_image(dot_file='tree.dot', output_image='tree.png'):
    """
    Executes a shell command to convert a Graphviz dot file to a PNG image file.

    Parameters:
        dot_file (str): The path to the dot file.
        output_image (str): The output image file path.
    """
    # Build the command as a list of parts
    command = ['dot', '-Tpng', dot_file, '-o', output_image]

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        print(f"Generated image successfully at '{output_image}'.")
    else:
        print(f"Failed to generate image. Error: {result.stderr}")
