# Machine Learning Classifiers Project
This project demonstrates five popular machine learning algorithms applied to well-known datasets using Python and scikit-learn. Each script loads a different dataset, trains a model, evaluates accuracy, and prints the result.

# 📂 Project Structure
| Script                     | Algorithm              | Dataset                                 |
| -------------------------- | ---------------------- | --------------------------------------- |
| `1_logistic_regression.py` | Logistic Regression    | Iris (from `sklearn.datasets`)          |
| `2_decision_tree.py`       | Decision Tree          | Titanic (from `seaborn`)                |
| `3_knn.py`                 | K-Nearest Neighbors    | Digits (from `sklearn.datasets`)        |
| `4_svm.py`                 | Support Vector Machine | Breast Cancer (from `sklearn.datasets`) |
| `5_random_forest.py`       | Random Forest          | Wine (from `sklearn.datasets`)          |

# 🔍 Descriptions
# 1. 1_logistic_regression.py
Algorithm: Logistic Regression
Dataset: Iris (150 samples, 3 species)

What it does:
- Loads Iris dataset using load_iris()
- Splits into training/testing sets
- Trains a LogisticRegression model
- Evaluates accuracy with accuracy_score

# 2. 2_decision_tree.py
Algorithm: Decision Tree Classifier
Dataset: Titanic dataset from kaggle 

What it does:
- Loads Titanic dataset 
- Preprocesses features (drops missing values, encodes gender)
- Trains a DecisionTreeClassifier
- Prints classification accuracy

# 3. 3_knn.py
Algorithm: K-Nearest Neighbors (KNN)
Dataset: Digits dataset 

What it does:
- Loads the dataset using load_digits()
- Splits data into train/test
- Uses KNeighborsClassifier with k=3
- Evaluates and prints test accuracy

# 4. 4_svm.py
Algorithm: Support Vector Machine (SVM)
Dataset: Breast Cancer dataset from kaggle

What it does:
- Loads dataset 
- Trains a SVC() (Support Vector Classifier)
- Predicts on test data and evaluates performance

# 5. 5_random_forest.py
Algorithm: Random Forest Classifier
Dataset: Wine dataset from kaggle

What it does:
- Loads Wine dataset 
- Trains a RandomForestClassifier with default parameters
- Predicts and prints the accuracy on the test set


