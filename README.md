#  Employee Salary Prediction Using Machine Learning Algorithms

This project predicts whether an employee's income exceeds $50K/year using machine learning techniques. The model is built using Python, Scikit-Learn, Pandas, and visualized with Seaborn and Matplotlib.



##  Problem Statement

The aim is to develop a predictive model that classifies whether an individual's income is `>50K` or `<=50K` based on demographic and professional attributes such as age, education, occupation, work hours, etc.

This is a classic **binary classification** problem and is widely used to demonstrate data preprocessing, feature engineering, and model evaluation techniques.


##  Architecture Overview

The solution is structured into the following stages:

1. **Data Loading**  
   Load dataset using `pandas`.

2. **Preprocessing**  
   - Label encode the target variable (`income`).
   - One-hot encode categorical features.
   - Scale numerical features using `StandardScaler`.
   - Use `ColumnTransformer` to process features in parallel.

3. **Modeling**  
   Use a `RandomForestClassifier` inside a `Pipeline` for automation and scalability.

4. **Evaluation**  
   - `Accuracy Score`  
   - `Classification Report`  
   - `Confusion Matrix` (with heatmap visualization)

5. **Visualization**  
   Confusion matrix is plotted using `seaborn.heatmap`.


##  Tech Stack

- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit (optional for deployment)

##  Dataset

- File: `employee_salary_prediction_dataset.csv`
- Target: `income` (binary: `>50K` or `<=50K`)
- Features include: age, workclass, education, occupation, hours-per-week, etc.
- Categorical + numerical features

##  Features Used

| Feature Name   | Type        |
|----------------|-------------|
| Age            | Numerical   |
| Workclass      | Categorical |
| Education      | Categorical |
| Marital Status | Categorical |
| Occupation     | Categorical |
| Capital Gain   | Numerical   |
| Hours per Week | Numerical   |
| Native Country | Categorical |


##  How to Run the Project

1. **Install dependencies**
   pip install pandas numpy matplotlib seaborn scikit-learn
   
2.**Run the Python script**

  python salary_prediction.py
  
3.**Optional Streamlit App**

pip install streamlit
streamlit run app.py

**Model Performance**
Model: RandomForestClassifier with 200 trees

Accuracy: ~85–90% (depending on random state)

Deployment Ready: Yes (Streamlit or Flask)

**Evaluation Metrics**

Accuracy Score
Precision, Recall, F1-score
Confusion Matrix with Heatmap
Stratified Train/Test Split (80:20)

 **Future Scope**
 Model Tuning: Use GridSearchCV or RandomSearchCV for hyperparameter optimization.

 Other Models: Try Gradient Boosting, XGBoost, LightGBM, etc.

 Web Deployment: Create a Streamlit or Flask web interface.

 AutoML: Integrate automated ML pipelines using PyCaret or H2O.ai.


![8](https://github.com/user-attachments/assets/889b54df-8b77-47f2-b500-4b91ad023b16)

**License**
This project is for educational and internship purposes. All rights reserved by the author.

   
