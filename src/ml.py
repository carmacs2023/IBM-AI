import pandas as pd
import numpy as np
# import pylab as pl
# import requests
# import sys
import os
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess


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
    y = df[target_col_name].values      # Target variable
    # x = df.drop(columns=[target_col_name]).values  # Remove the target column for feature set
    # y = df[target_col_name].values  # Target variable

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
        # K-Nearest Neighbors Classification
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train, y_train)  # Training the model

        # Predicting the test set results
        yhat = neigh.predict(x_test)
        print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train)))
        print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


def decision_tree_classifier(df: pd.DataFrame, target_col_name: str, test_size: float, random_state: int, criterion: str = "entropy",
                             max_depth: int = None, dot_filename: str = "tree.dot", label_encoder_dict: dict = None):
    """
    Train and visualize a decision tree classifier.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col_name (str): The name of the target variable column.
        criterion (str): The function to measure the quality of a split. Default is "entropy".
        max_depth (int): The maximum depth of the tree.
        dot_filename (str): The filename for the exported dot file.
        label_encoder_dict (dict): Dictionary of column names and their respective LabelEncoder() instances.
            Sklearn Decision Trees does not handle categorical variables.
            Use LabelEncoder() method to convert these features to numerical values.
            for example Low High to 0 and 1.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator for reproducibility.

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

    print(df.head())
    print(df.shape)
    print(df[target_col_name].value_counts())
    print(f'Before using label encoder:\n {df.drop(target_col_name, axis=1).values[0:5]}')

    # Encode categorical variables using LabelEncoder
    for column, encoder in label_encoder_dict.items():
        df[column] = encoder.fit_transform(df[column])

    # Splitting the dataset into features and target variable
    x = df.drop(target_col_name, axis=1).values

    print(f'After using label encoder:\n {x[0:5]}')

    y = df[target_col_name].values

    # Splitting the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Modeling and Training the Decision Tree
    drug_tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    drug_tree.fit(x_train, y_train)

    # Predicting the test set results
    y_pred = drug_tree.predict(x_test)

    # Evaluating the model
    print("Decision Tree Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    # Export the trained model to a dot file
    export_graphviz(drug_tree, out_file=dot_filename, filled=True, feature_names=df.drop(target_col_name, axis=1).columns.tolist())
    generate_tree_image(dot_file=dot_filename, output_image=dot_filename.replace('.dot', '.png'))

    # Open the tree image; adjust the command according to your OS
    os.system(f'open {dot_filename.replace(".dot", ".png")}')

