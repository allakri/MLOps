# Wine Quality Prediction using Hyperparameter Optimization

This project demonstrates how to use Keras for training a neural network model on the **Wine Quality** dataset and optimize its hyperparameters using **Hyperopt**. Additionally, we use **MLflow** to log experiments, track models, and store the best-performing configurations.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Libraries](#libraries)
- [Model Architecture](#model-architecture)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Usage](#usage)
- [Output](#output)

## Overview

This project involves training a neural network to predict the quality of white wine based on various chemical attributes. We use **Hyperopt** to search for the best hyperparameters for the model (learning rate and momentum) and **MLflow** for experiment tracking, model logging, and performance evaluation.

## Dataset

The dataset used for this project is the **Wine Quality Dataset** from UCI's Machine Learning Repository, which contains several physicochemical properties of white wine, such as acidity, alcohol content, and more, along with a target variable `quality` (a score from 0 to 10).

- **Data Source**: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **Columns**:
  - `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`
  - Target: `quality`

The dataset is split into **training (75%)**, **validation (20%)**, and **test (25%)** sets.

## Libraries

- **Keras**: Deep learning library used for building and training the neural network model.
- **Numpy**: Library for numerical operations, handling arrays and matrices.
- **Pandas**: Data manipulation and analysis library for loading and working with the dataset.
- **Hyperopt**: Optimization library used to perform hyperparameter tuning of the model.
- **Scikit-learn**: Machine learning library used for dataset splitting and evaluation.
- **MLflow**: Experiment tracking tool used to log hyperparameters, metrics, and models.

## Model Architecture

The model used in this project is a simple feed-forward neural network built using Keras. The architecture includes:

- **Normalization Layer**: Standardizes the input features to improve convergence.
- **Dense Layers**: Two fully connected layers with ReLU activation in the hidden layer and a linear output layer for regression.

The model is compiled with **Stochastic Gradient Descent (SGD)** optimizer, using **learning rate (lr)** and **momentum** as hyperparameters, and optimized using **Mean Squared Error (MSE)** loss with **Root Mean Squared Error (RMSE)** as the evaluation metric.

### Model Training Process

The model is trained using the **training set**, with validation on the **validation set**. After each training run, the model is evaluated on the validation set, and the **RMSE** is calculated to assess performance.

## Hyperparameter Optimization

We perform hyperparameter optimization using **Hyperopt**. The objective is to find the best combination of the following hyperparameters:

- **Learning rate (`lr`)**: Sampled from a **log-uniform** distribution between `1e-5` and `1e-1`.
- **Momentum (`momentum`)**: Sampled from a **uniform** distribution between `0.0` and `1.0`.

The **fmin** function from Hyperopt is used to minimize the **RMSE** loss on the validation set over multiple trials. The best set of hyperparameters is chosen based on the minimum validation RMSE.

## MLflow Experiment Tracking

We use **MLflow** to:

- Log hyperparameters (`lr`, `momentum`) and metrics (`eval_rmse`).
- Track the best-performing models based on RMSE.
- Store the model architecture and parameters.
- Keep track of multiple runs and visualize experiment progress.

Each experiment is logged under the `wine-quality` experiment name in MLflow. The signature of the model (input and output schema) is inferred from the training data and stored with the model for future use.

## Usage

### Requirements

Before running the code, make sure you have the following libraries installed:

```bash
pip install keras numpy pandas hyperopt scikit-learn mlflow
```

### Running the Code

1. Load the dataset and split it into training, validation, and test sets.
2. Define the neural network model using Keras.
3. Perform hyperparameter optimization using Hyperopt to find the best combination of `learning rate` and `momentum`.
4. Log experiments using MLflow to track model performance, hyperparameters, and the final trained model.

The training script automatically tracks and logs the experiments, hyperparameters, and models to **MLflow**. After running, the best hyperparameters, model, and evaluation metrics will be saved.

### Example Command to Run the Experiment:

```python
python train_model.py
```

## Output

- **Best Hyperparameters**: The learning rate and momentum values that result in the best RMSE on the validation set.
- **Best Model**: The neural network model trained with the best hyperparameters.
- **Logs**: The hyperparameters, RMSE, and trained model are logged in **MLflow** under the `/wine-quality` experiment.

### Example of logged results:

- **Best Parameters**:
  - `lr: 0.0012`
  - `momentum: 0.8`
- **Best Evaluation RMSE**: `0.59`
- **Trained Model**: The model will be saved in MLflow for future use or deployment.

## Conclusion

This project provides an example of using **Keras**, **Hyperopt**, and **MLflow** to build and optimize a neural network model for predicting wine quality. It demonstrates how to automate hyperparameter tuning, log experiments, and manage models in a reproducible way.
