# Customer-Churn-Prediction
Project Title: Customer Churn Prediction

Description: This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. The dataset utilized is the Telco Customer Churn dataset, which contains various features related to customer demographics, account information, and service usage. The primary objective is to build a predictive model that can identify customers likely to churn, enabling the company to take proactive measures to retain them.

Key Components:

1. Data Loading and Preprocessing:
The dataset is loaded using pandas from a CSV file, resulting in a DataFrame with 7043 entries and 21 columns.
The customerID column is dropped as it does not contribute to the predictive analysis.
The TotalCharges column, initially in string format, is converted to float after replacing empty values with 0.0 to ensure proper numerical analysis.
The Churn column is transformed from categorical values ('Yes', 'No') to binary values (1 for 'Yes', 0 for 'No') for easier model training.

2. Exploratory Data Analysis (EDA):
The code performs an initial exploration of the dataset, checking for null values and unique values in categorical columns.
Histograms and box plots are generated for numerical features (tenure, MonthlyCharges, TotalCharges) to visualize their distributions:
Tenure: Shows the duration of customer subscriptions.
Monthly Charges: Displays the monthly billing amounts.
Total Charges: Represents the total amount billed to customers.
A correlation heatmap is created to understand the relationships between numerical features, indicating how features like tenure, MonthlyCharges, and TotalCharges correlate with each other.

3. Data Encoding:
Categorical features are encoded using LabelEncoder to convert them into numerical format suitable for machine learning models.
Encoders for each categorical column are saved to a file for future use, ensuring that the same encoding can be applied to new data.

4. Data Splitting:
The dataset is split into training and testing sets using train_test_split, with 80% of the data allocated for training and 20% for testing.
The training set consists of 5634 samples, while the test set contains 1409 samples.

5. Handling Class Imbalance:
The training data is balanced using SMOTE (Synthetic Minority Over-sampling Technique) to address the class imbalance in the target variable (Churn).
After applying SMOTE, the training set is balanced to have 4138 instances of both classes (0 and 1).

6. Model Training:
Three machine learning models are defined:
Decision Tree
Random Forest
XGBoost
Each model is trained using cross-validation to evaluate its performance based on accuracy. The results are as follows:
Decision Tree: Cross-validation accuracy of approximately 78%.
Random Forest: Cross-validation accuracy of approximately 84%.
XGBoost: Cross-validation accuracy of approximately 83%.

7. Model Evaluation:
The Random Forest model is selected for final evaluation due to its superior performance.
Predictions are made on the test set, and performance metrics are generated:
Accuracy Score: 77.86%
Confusion Matrix:

[[878 158]
 [154 219]]

Classification Report:

            precision    recall  f1-score   support

         0       0.85      0.85      0.85      1036
         1       0.58      0.59      0.58       373

  accuracy                           0.78      1409
 macro avg       0.72      0.72      0.72      1409

weighted avg 0.78 0.78 0.78 1409

8. Model Saving:
The trained Random Forest model and its feature names are saved to a file (customer_churn_model.pkl) for future predictions.

9. Making Predictions:
A sample input data is prepared, representing a hypothetical customer.
The model is used to predict whether the customer will churn or not, along with the prediction probability.
Prediction Result: The model predicts "No Churn" with a probability of 78% for the customer.

10. Downsampling (Optional):
An alternative approach to handle class imbalance is demonstrated by downsampling the majority class to match the minority class size.
After downsampling, the class distribution is confirmed to be balanced with 1869 instances of each class.

11. Hyperparameter Tuning:
A grid search is performed to find the best hyperparameters for the Random Forest model, optimizing for accuracy using cross-validation.
Best Parameters Found:
max_depth: 5
n_estimators: 100
Best Cross-Validation Score: 76.88%

Conclusion:
This project provides a comprehensive approach to customer churn prediction, from data preprocessing and exploratory analysis to model training and evaluation. The insights gained from this analysis can help the telecommunications company implement strategies to reduce churn and improve customer retention. The Random Forest model, with its high accuracy and interpretability, serves as a valuable tool for predicting customer behavior and informing business decisions.
