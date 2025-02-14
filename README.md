# Heart-attack-risk-prediction-machine-learning
![image](https://github.com/user-attachments/assets/9c81af1c-4193-497a-9c47-e75e46e121fb)

 Early prediction of heart attack risk can significantly improve patient outcomes. This project aims to build a machine learning model to predict the risk of heart attack based on age, gender, cholesterol levels, blood pressure, heart rate, and indicators like diabetes, family history, smoking habits, obesity, and alcohol consumption.

The dataset from [kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset) contains 8,763 records from pateints around the world with 26 features. The target variable (heart attack risk) is binary either 0= no risk, or 1= risk. The project involves data preprocessing and implementation of KNN (K-Nearest Neighbors) to predict heart attack risk. 

# Steps:

## Data Preprocessing:

Dropped irrelevant columns (Country, Continent, Hemisphere).
Encoded categorical variables (Sex, Diet) using ordinal mapping.
Split the Blood Pressure column into BP_Systolic and BP_Diastolic for better analysis.
Handled missing values (none found in this dataset).

## Exploratory Data Analysis (EDA):

Used pandas and seaborn to analyze and visualize the dataset.
Checked data distribution and correlations.
## Model Building:
Split the data into training and testing sets using train_test_split.
Applied K-Nearest Neighbors (KNN) and Random Forest classifiers.


## Model Evaluation:

Evaluated models using confusion matrix, accuracy score, and classification report.
Achieved high accuracy with both KNN 

## Visualization:
Created a heatmap to visualize the confusion matrix.

## Libraries Used:
numpy, pandas, matplotlib, seaborn for data manipulation and visualization.
scikit-learn for machine learning KNN.
imbalanced-learn for handling class imbalance.

### Results:
KNN Model: Achieved an accuracy of 97.43%.

![image](https://github.com/user-attachments/assets/87fc6769-40de-4b01-9b80-d33ff47eec09)

![image](https://github.com/user-attachments/assets/fce5fcea-ba9b-45ae-beaa-a0c710d3c43c)

568= TP, CORRECTLY PREDICTED TO BE AT RISK OF HAVING HEART ATTACK :(

1140 = TN, CORRECTLY PREDICATED TO NOT BE AT RISK OF HEART ATTACK :)

2= FP, FALSLY PREDICATED TO HAVE RISK OF HEART ATTACK

43= FN, FALSLY PREDICTED TO HAVE NO RISK OF HEART ATTACK

