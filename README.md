# Employee Salary Prediction Model

A machine learning project that predicts employee salaries based on features such as age, gender, education level, job title, and years of experience. Built as a capstone project to help HR departments and companies make fair, data-driven compensation decisions by learning patterns from historical salary data.

## Overview

Salary decisions are often inconsistent across companies due to manual or biased estimation. This project builds a regression-based ML pipeline that learns from historical employee data to predict fair salary estimates, promoting transparency and equity in compensation.

## Features

- Predicts salary based on age, gender, education level, job title, and years of experience
- Compares multiple regression models and selects the best performer
- Interactive Streamlit web app for real-time salary predictions
- Data visualizations (correlation heatmaps, distribution plots, pair plots) for exploratory analysis

## Tech Stack

- **Language:** Python 3.6+
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Statistical Analysis:** Statsmodels
- **Deployment:** Streamlit

## Project Workflow

1. **Load Data** — Import the dataset into a Pandas DataFrame.
2. **Explore Data** — Check shape, columns, data types, duplicates, and missing values.
3. **Clean Data** — Remove duplicate rows and drop rows with missing values.
4. **Visualize Data** — Generate bar plots, histograms, and correlation heatmaps to understand feature relationships.
5. **Transform Data**
   - Encode categorical features using `LabelEncoder`
   - Scale numerical features using `StandardScaler`
6. **Split Data** — Divide the dataset into training and testing sets.
7. **Train Models**
   - Linear Regression
   - Random Forest Regressor
   - Voting Regressor (ensemble of Linear Regression + Random Forest)
8. **Evaluate Models** — Compare models using R², MAE, and MSE.
9. **Save Artifacts** — Persist the best-performing model (Voting Regressor) along with fitted encoders and scaler using Joblib.
10. **Deploy** — Serve predictions through a Streamlit app (`app.py`) that loads the saved model and transformers to process user input.

## Results

- Multiple regression models were trained and compared using R², MAE, and MSE.
- The **Voting Regressor** was selected as the best-performing model, offering strong predictive accuracy with low error.
- Exploratory visualizations (correlation heatmap, salary distribution, education-level counts, pair plots) supported feature understanding and preprocessing decisions.

## Requirements

### Hardware
- Processor: Intel i5 or equivalent
- RAM: 8 GB minimum
- Storage: 10 GB minimum

### Software
- OS: Windows, macOS, or Linux
- Python 3.6+
- IDE: Jupyter Notebook, PyCharm, or VS Code

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/SHUBHAM-SHINDE-2003/Employee-Salary-Prediction-Model.git
cd Employee-Salary-Prediction-Model

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib statsmodels streamlit

# Run the Streamlit app
streamlit run app.py
```

## Challenges Encountered

- **Data quality issues** — Handling missing values and duplicates to maintain data integrity.
- **Feature engineering** — Ensuring categorical encoding was applied consistently across training and testing sets.
- **Model selection & tuning** — Extensive experimentation to identify the best-performing algorithm and hyperparameters.
- **Deployment** — Correctly loading the saved model and preprocessing steps within the Streamlit app for live user input.

## Future Scope

- Expand the dataset with more diverse, industry-specific data to improve generalization.
- Apply advanced feature engineering (e.g., one-hot encoding, polynomial features).
- Improve the Streamlit UI for a more intuitive user experience.

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly Media.
- Streamlit Documentation.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.

## Author

**Shubham S. Shinde**
Information Technology Department, MGM'S College Of Engineering, Nanded
