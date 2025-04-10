
# Arthur Movsesyan
# CISC 3440: Machine Learning
# P2
# We work with the Kaggle Dataset (https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024) for this project. The dataset involves predicting if a nearest object to earth is Hazardous.
# 
# The project will involve two parts:
# 1. In the first part we implemented the Perceptron Algorithm and Train the model using the data
# 2. In the second part we used Linear Regression to build a model
# 
# 
# We used the file train.csv and used Pandas to load and process the feature.
# 
# **Data Features:** Each line in the file has the following Features
# 
# | **Column Name**             | **Description**                                                                 |
# |-----------------------------|---------------------------------------------------------------------------------|
# | `absolute_magnitude`        | The absolute magnitude of the object, indicating its brightness.               |
# | `estimated_diameter_min`    | The minimum estimated diameter of the object, likely in kilometers or meters.  |
# | `estimated_diameter_max`    | The maximum estimated diameter of the object, likely in kilometers or meters.  |
# | `relative_velocity`         | The speed at which the object is moving relative to Earth, typically in m/s or km/h. |
# | `miss_distance`             | The minimum distance by which the object is expected to miss Earth, possibly in kilometers. |
# | `is_hazardous`              | A boolean indicator (TRUE/FALSE) of whether the object is classified as potentially hazardous to Earth. |
# 

# # Part-1 Perceptron
# 
# For the first part we implemented the Perceptron Algorithm, which we use to train the data.
# 
# **Perceptron Algorithm**
# 
# The Perceptron algorithm will take the following parameters as input:
# - training_samples of shape (N,D) where N is the number of samples and D is the number of features
# - training_labels of shape (N,1) where N is the number of samples
# - iterations the number of times to iterate over the entire dataset
# 
# The function returns two lists:
# 
# - *weight_values* : A list of weight vectors obtained at the end of each iteration, this list will contain elements equivalent to the number of iterations
# 
# - *bias_values* : A list of bias values obtained at the end of each iteration, this list will also contain elements equivalent to the number of iterations

import numpy as np
import matplotlib.pyplot as plt
import random

def perceptron_algorithm(training_samples, training_labels, iterations):
    """
    Trains a Perceptron model on the provided training samples and labels for a specified number of iterations.

    Parameters
    ----------
    training_samples : numpy.ndarray
        A 2D array of shape (num_samples, num_features), where each row represents a training sample
        and each column represents a feature.

    training_labels : numpy.ndarray
        A 1D array of shape (num_samples,), where each element is the label for the corresponding
        training sample in `training_samples`.

    iterations : int
        The number of epochs (full passes through the training data) to perform.

    Returns
    -------
    weight_values : list of numpy.ndarray
        A list of numpy arrays, each representing the weights after each epoch.

    bias_values : list of float
        A list of bias values after each epoch.

    """
    weight_values = []
    bias_values = []
    
    # Initialize weights and bias
    num_features = training_samples.shape[1]
    weights = np.random.randn(num_features)
    bias = random.uniform(-1, 1)

    for _ in range(iterations):
        for i in range(len(training_samples)):
            # Calculate the activation
            activation = np.dot(weights, training_samples[i]) + bias
            
            # Update weights and bias if misclassified
            if training_labels[i] * activation <= 0:
                weights += training_labels[i] * training_samples[i]
                bias += training_labels[i]
        
        # Append weights and bias for this epoch
        weight_values.append(weights.copy())
        bias_values.append(bias)
    
    return weight_values, bias_values


# **Predictions**


def perceptron_predict(test_sample,w,b):
  """
    Predicts the class label for a given test sample using the Perceptron model.


    Parameters
    ----------
    test_sample : numpy.ndarray
        A 1D array representing the feature values of the test sample.

    w : numpy.ndarray
        A 1D array representing the weights learned by the Perceptron algorithm.

    b : float
        The bias term learned by the Perceptron algorithm.

    Returns
    -------
    int
        The predicted class label for the test sample: +1 if the sample is classified as positive,
        or -1 if classified as negative.
  """
  prediction=None

  # Calculate the activation
  activation = np.dot(w, test_sample) + b

  # Determine the prediction based on the activation

  if activation > 0:
    prediction = 1
  else:
    prediction = -1

  return prediction


# ### Data Loading and Preprocessing


import pandas as pd

trainData=pd.read_csv("train.csv")
print(trainData.info())


# **Cleaning and Preprocessing**

features=None
labels=None

from sklearn.preprocessing import StandardScaler


#checking for null values
print(f"Missing value check: {trainData.isnull().sum()}" )

print("Filling in numm values with -999")
trainData.fillna(-999, inplace=True)

print("Checking if it still contains null values")
print(trainData.isnull().sum())

features= trainData[["absolute_magnitude", "estimated_diameter_min", "estimated_diameter_max", "relative_velocity", "miss_distance"]].copy()
labels=trainData[["is_hazardous"]].replace({False: -1, True : 1})

#Standarize the features
scaler = StandardScaler()
features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)

print(features.head())
print(labels.head())




# **Training and validation sets**
# 
# We use the *train_test_split()* method to create training and validation sets. Retaining 90% of the data for training and 10% for testing.
# 
# The perceptron algorithm will take numpy arrays as input so we convert our split data into numpy arrays using if the resulting arrays after the split are not numpy arrays.
# 
# > df.to_numpy()
# 
# where df is our data frame.
# 
# By the end of this, we have four variables :
# 
# - train_features_numpy - Numpy array of dimension NxD where N is the number of    samples
# - train_labels_numpy - Numpy array of dimension Nx1
# - val_features_numpy - Numpy array of dimension MxD where M is the number of samples
# - val_labels_numpy - Numpy array of dimension Mx1


from sklearn.model_selection import train_test_split


#Convert features and labels to numpy arrays
features_numpy = features.to_numpy()
labels_numpy = labels.to_numpy()


#Splitting the data into training and validating
train_features_numpy, val_features_numpy, train_labels_numpy, val_labels_numpy = train_test_split(features_numpy, labels_numpy, test_size=0.1, random_state=42)
#train_size= 0.9, test_size = 0.1 for 10% validating, random state = 42 for reproducibility

print("Shapes of the data")
print(train_features_numpy.shape)
print(train_labels_numpy.shape)
print(val_features_numpy.shape)
print(val_labels_numpy.shape)

print("Type of the data")
print(train_features_numpy.dtype)
print(train_labels_numpy.dtype)
print(val_features_numpy.dtype)
print(val_labels_numpy.dtype)


# **Training and validation**
# 
# We train the model for 50-100 iterations. Aftwards, we use the returned weights and bias values to make predictions on the validation data and compute accuracy and F1-score for the weights and bias values from every iteration.
# 
# For computing accuracy and f1_score we use sklearn library.
# 
# - Computing accuracy using sklearn - refer [here](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)
# 
# - Computing F1-score using sklearn - refere [here](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
# 
# The accuracy value should be stored in the variable name *accuracy_over_iterations* and F1-score in the variable name *f1_score_over_iterations*. These values will be used to generate plot which for now uses randomly generate accuracy and F1-scores for 100 iterations.


iterations=50 #number of epochs
w,b=perceptron_algorithm(train_features_numpy,train_labels_numpy.reshape(-1),iterations)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score

# accuracy_over_iteations=np.random.rand(100) # You should comment this once you have made your predictions
# f1_score_over_iteations=np.random.rand(100) # You should comment this once you have made your predictions


accuracy_over_iterations = []
f1_score_over_iterations = []

for i in range(iterations):
    val_predictions = []
    for sample in val_features_numpy:
        prediction = perceptron_predict(sample, w[i], b[i])
        val_predictions.append(prediction)

    val_predictions = np.array(val_predictions).reshape(-1, 1)  # Reshape to match val_labels_numpy

    accuracy_over_iterations.append(accuracy_score(val_labels_numpy, val_predictions))
    f1_score_over_iterations.append(f1_score(val_labels_numpy, val_predictions, pos_label=1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(accuracy_over_iterations, label='Accuracy', marker='o', alpha=0.5)
plt.plot(f1_score_over_iterations, label='F1-score', marker='o', alpha=0.5)
plt.title("Perceptron Performance Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.ylim([0, 1])
plt.show()


# **Balancing the Dataset**
# 
# The distribution of +1 and -1 samples are imbalanced in the dataset. Which means either:
# 
# - Undersample
# - Oversample
# 
# The code below shows how to undersample it and stores the resulting data into the variable *df_balanced*. We generate df_balanced to generate new features and labels and repeat the training process.
# 
# 
# 


majority_class = trainData[trainData['is_hazardous'] == False]
minority_class = trainData[trainData['is_hazardous'] == True]

#printing original distribution
print("Original class distribution")
print("Majority class: " , len(majority_class))
print("Minority class: " , len(minority_class))


majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)
df_balanced = pd.concat([majority_downsampled, minority_class])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

#Printing balanced distribution
print("Balanced class distribution")
print("Majority class: " , len(df_balanced[df_balanced['is_hazardous'] == False]))
print("Minority class: " , len(df_balanced[df_balanced['is_hazardous'] == True]))


df_balanced.info()

#Seperating the feautre and labels from balanced data
features=df_balanced[['absolute_magnitude', 'estimated_diameter_min',
                       'estimated_diameter_max', 'relative_velocity',
                       'miss_distance']].copy()


labels= df_balanced['is_hazardous'].replace({False: -1, True : 1})

#Standarize the features
scaler = StandardScaler()
features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)


from sklearn.model_selection import train_test_split

features_numpy = features.to_numpy()
labels_numpy = labels.to_numpy().reshape(-1, 1)

train_features_numpy, val_features_numpy, train_labels_numpy, val_labels_numpy = train_test_split(features_numpy, labels_numpy, test_size=0.1, random_state=42)


print("Training features:", train_features_numpy.shape)
print("Training labels:", train_labels_numpy.shape)
print("Validation features:", val_features_numpy.shape)
print("Validation labels:", val_labels_numpy.shape)



iterations=20
w,b=perceptron_algorithm(train_features_numpy,train_labels_numpy,iterations)


from sklearn.metrics import f1_score,accuracy_score

# accuracy_over_iteations=np.random.rand(100) # You should comment this once you have made your predictions
# f1_score_over_iteations=np.random.rand(100) # You should comment this once you have made your predictions

accuracy_over_iterations = []
f1_score_over_iterations = []

for i in range(iterations):
    val_predictions = []
    for sample in val_features_numpy:
        prediction = perceptron_predict(sample, w[i], b[i])
        val_predictions.append(prediction)
    # Reshape to match val_labels_numpy
    val_predictions = np.array(val_predictions).reshape(-1, 1)

    accuracy_over_iterations.append(accuracy_score(val_labels_numpy, val_predictions))
    f1_score_over_iterations.append(f1_score(val_labels_numpy, val_predictions, pos_label=1))

plt.figure(figsize=(10, 6))
plt.plot(accuracy_over_iterations, label='Accuracy', marker='o', alpha=0.5)
plt.plot(f1_score_over_iterations, label='F1-score', marker='o', alpha=0.5)
plt.title("Perceptron Performance on Balanced Dataset")
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.ylim([0, 1])
plt.show()


import pickle
chosen_iteration=None

if chosen_iteration is not None:
  weight=w[chosen_iteration]
  bias=b[chosen_iteration]
  model=[weight,bias]
  with open('perceptron_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# ## Part-II Linear Regression
# 
# For this part, we use the same balanced dataset df_balanced that we previously created and fit a linear regression model. 
# 
# We interepret the class based on the following condition:
#   > If the prediction is > 0 you will label it as +1
#   > If the prediction is < 0 you will label it as -1
# 
# We use the predictions of the validation set mapped to +1 and -1 to compute tha accuracy and F1_score.
# 

import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Inspect the Data
print("Columns in df_balanced:", df_balanced.columns)
print("Sample data from df_balanced:")
print(df_balanced.head())


target_column = "is_hazardous"


if target_column not in df_balanced.columns:
    raise KeyError(f"Target column '{target_column}' not found in df_balanced.")

df_balanced[target_column] = df_balanced[target_column].apply(lambda x: 1 if x else -1)

# Step 2: Prepare the data
X = df_balanced.drop(columns=[target_column])  
y = df_balanced[target_column]  

# Step 3: Train-validation split (90-10 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Step 4: Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_val)
lr_predictions_mapped = np.where(lr_predictions > 0, 1, -1)

lr_accuracy = accuracy_score(y_val, lr_predictions_mapped)
lr_f1 = f1_score(y_val, lr_predictions_mapped)

print(f"Linear Regression - Accuracy: {lr_accuracy}, F1 Score: {lr_f1}")

# Step 5: Train Ridge regression models with different alpha values
ridge_results = []
alpha_values = [0.01, 0.1, 1, 10, 100]

for alpha in alpha_values:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    ridge_predictions = ridge_model.predict(X_val)
    ridge_predictions_mapped = np.where(ridge_predictions > 0, 1, -1)
    ridge_accuracy = accuracy_score(y_val, ridge_predictions_mapped)
    ridge_f1 = f1_score(y_val, ridge_predictions_mapped)
    ridge_results.append((alpha, ridge_accuracy, ridge_f1, ridge_model))


if not ridge_results:
    raise ValueError("Ridge regression models were not trained. Check your training loop and data.")

best_ridge = max(ridge_results, key=lambda x: x[2])  
best_alpha, best_accuracy, best_f1, best_model = best_ridge

print(f"Ridge Regression - Best Alpha: {best_alpha}, Accuracy: {best_accuracy}, F1 Score: {best_f1}")

# Step 6: Save the best model
final_model = best_model

print("Final model (best_model):", final_model)
print("Model type:", type(final_model))



print("Saving model to 'linear_regression_model.pkl'...")
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)


file_path = 'linear_regression_model.pkl'
if os.path.exists(file_path):
    print(f"File '{file_path}' successfully created. Size: {os.path.getsize(file_path)} bytes")
else:
    print(f"File '{file_path}' was not created.")

