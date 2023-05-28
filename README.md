
# Heart Disease Prediction
This project focuses on using machine learning to predict whether a person is suffering from heart disease. The dataset used in this project is obtained from Kaggle (link to dataset).

## Project Overview

### Data Exploration: 
Load the dataset and perform initial data exploration using pandas and matplotlib to gain insights into the data.

### Feature Selection: 
Analyze the correlation between features and select relevant features for the prediction model.

### Data Processing: 
Preprocess the dataset by converting categorical variables into dummy variables and scaling numerical features using StandardScaler.

### Model Selection and Evaluation:
K-Nearest Neighbors (KNN): Experiment with different values of K and evaluate the performance using cross-validation.

Decision Tree Classifier: 
Implement a decision tree-based classification model.

Random Forest Classifier: Build a random forest model and evaluate its performance using cross-validation.

### Results: 
Summarize the performance of each model and choose the best-performing model for heart disease prediction.

# Getting Started
To get started with this project, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/heart-disease-prediction.git
Install the required dependencies:

Copy code
pip install -r requirements.txt
Run the Jupyter Notebook or Python script to execute the project pipeline.

## Dependencies
numpy
pandas
matplotlib
seaborn
scikit-learn

## Results and Discussion
Based on the cross-validation scores obtained, the K-Nearest Neighbors (KNN) model with K=12 achieved an accuracy of approximately 85%. The Random Forest Classifier achieved an accuracy of around 82%.

The project demonstrates the application of machine learning algorithms to predict heart disease. Further improvements can be made by experimenting with different algorithms or hyperparameter tuning to enhance the performance.

## Conclusion
In this project, a machine learning model was developed to predict heart disease. The dataset was preprocessed, and three different classification algorithms were implemented and evaluated. The KNN model with K=12 achieved the highest accuracy.

Feel free to contribute to this project by adding new features, exploring different algorithms, or enhancing the model's performance.

