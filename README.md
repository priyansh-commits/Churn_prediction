# 📊 Telecom Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn based on demographic, service, contract, and billing information.

The project covers the complete machine learning workflow — from exploratory data analysis and preprocessing to model training, evaluation, model saving, and deployment using Streamlit.

## 🚀 Live Demo

**Streamlit App:** (https://8p82bdtcf2qhfn6p6vmzur.streamlit.app/)

## 📌 Project Overview

Customer churn prediction helps telecom companies identify customers who are likely to leave their service.

In this project, I analyzed a telecom customer dataset and built multiple classification models to predict customer churn.

The workflow includes:

* Data cleaning
* Exploratory Data Analysis (EDA)
* Outlier and distribution analysis
* Categorical feature encoding
* Class imbalance analysis
* Correlation analysis
* Model training
* Model comparison
* Cross-validation
* ROC-AUC evaluation
* Model serialization using Joblib
* Streamlit deployment

## 📂 Dataset

The project uses the **Telecom Customer Churn** dataset from Kaggle.

The dataset contains customer information such as:

* Age
* Tenure
* Monthly Charges
* Total Charges
* Tech Support
* Gender
* Contract Type
* Internet Service
* Churn

### Target Variable

`Churn`

* `0` → No Churn
* `1` → Churn

The dataset is imbalanced, with approximately:

* **88.3% Churn**
* **11.7% No Churn**

Therefore, accuracy alone was not used to evaluate the models.

## 🔍 Exploratory Data Analysis

Several aspects of the dataset were investigated:

* Missing values
* Duplicate records
* Feature distributions
* Outliers
* Categorical variables
* Correlations with the target
* Class imbalance

Some observations:

* `Age` is approximately normally distributed.
* `Tenure` and `TotalCharges` are right-skewed.
* `Tenure` and `TotalCharges` show a strong positive relationship.
* Customers without Tech Support show a very strong association with churn.
* Contract type and tenure also show noticeable relationships with churn.

## ⚙️ Data Preprocessing

The preprocessing pipeline included:

1. Removing duplicate records.
2. Handling missing values.
3. Removing irrelevant features such as Customer ID.
4. Analyzing outliers and distributions.
5. Encoding categorical variables using one-hot encoding.
6. Separating features and target.
7. Splitting the data into training and testing sets.
8. Handling class imbalance during model experimentation.

The final model uses the following features:

```text
Age
Tenure
MonthlyCharges
TotalCharges
TechSupport
Gender_Male
ContractType_One-Year
ContractType_Two-Year
InternetService_Fiber Optic
InternetService_Nan
```

## 🤖 Models

The following classification algorithms were evaluated:

### Logistic Regression

Used as the baseline classification model.

### Decision Tree

Used to capture nonlinear relationships between customer characteristics and churn.

### Random Forest

Used as the final model because of its strong performance on the test data.

## 📊 Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |    0.955 |      0.95 |   0.95 |     0.95 |   0.985 |
| Decision Tree       |   ~0.995 |     ~0.99 |  ~0.99 |    ~0.99 |  ~0.997 |
| Random Forest       |    ~0.99 |     ~1.00 |  ~0.99 |    ~0.99 |    1.00 |

> **Note:** The dataset produces unusually strong model performance. The results should therefore be interpreted in the context of the dataset and its feature-target relationships rather than assumed to represent real-world telecom churn performance.

## 📈 Random Forest Evaluation

The Random Forest achieved approximately:

```text
Accuracy   ≈ 99%
ROC-AUC    = 1.00
```

Test-set classification results:

```text
              precision    recall  f1-score   support

0                1.00      0.96      0.98        23
1                0.99      1.00      1.00       177

accuracy                           0.99       200
macro avg         1.00      0.98      0.99       200
weighted avg      1.00      0.99      0.99       200
```

Cross-validation also produced consistently high ROC-AUC scores.

## 💾 Model Saving

The trained Random Forest model was serialized using Joblib:

```python
joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist()
    },
    "churn_model.pkl"
)
```

The saved model is loaded by the Streamlit application for making predictions.

## 🌐 Streamlit Application

The trained model was deployed using Streamlit.

The application allows users to enter customer information such as:

* Age
* Tenure
* Monthly Charges
* Total Charges
* Gender
* Tech Support
* Contract Type
* Internet Service

The application then provides:

* Churn prediction
* Churn probability

Example:

```text
Churn Probability: 97.5%

⚠️ Customer is likely to churn
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter / Google Colab
* GitHub

## 📁 Project Structure

```text
telecom-churn-prediction/
│
├── customer_churn.ipynb
├── churn_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd telecom-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 🔮 Future Improvements

Possible improvements include:

* Better handling of class imbalance
* Hyperparameter tuning
* Feature engineering
* Probability calibration
* More robust validation
* Testing on additional churn datasets
* Explainable AI using SHAP
* Improving the Streamlit UI
* Adding customer-level feature explanations

## 👨‍💻 Author

**Priyansh Khetarpal**

Electronics & Communication Engineering
Delhi Technological University (DTU)

---

⭐ If you found this project useful, feel free to explore the code and provide feedback.
