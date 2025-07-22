import streamlit as st
import pandas as pd
import joblib

# Load your model and preprocessing tools
try:
    model = joblib.load('voting_regressor_model.pkl')  
    le_gender = joblib.load('label_encoder_gender.pkl')
    le_education = joblib.load('label_encoder_education.pkl')
    le_job_title = joblib.load('label_encoder_jobtitle.pkl')
    scaler = joblib.load('standard_scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or preprocessing tools: {str(e)}")
    st.stop()

# Comprehensive list of job titles
JOB_TITLES = [
    'Software Engineer', 'Data Analyst', 'Senior Manager',
       'Sales Associate', 'Director', 'Marketing Analyst',
       'Product Manager', 'Sales Manager', 'Marketing Coordinator',
       'Senior Scientist', 'Software Developer', 'HR Manager',
       'Financial Analyst', 'Project Manager', 'Customer Service Rep',
       'Operations Manager', 'Marketing Manager', 'Senior Engineer',
       'Data Entry Clerk', 'Sales Director', 'Business Analyst',
       'VP of Operations', 'IT Support', 'Recruiter', 'Financial Manager',
       'Social Media Specialist', 'Software Manager', 'Junior Developer',
       'Senior Consultant', 'Product Designer', 'CEO', 'Accountant',
       'Data Scientist', 'Marketing Specialist', 'Technical Writer',
       'HR Generalist', 'Project Engineer', 'Customer Success Rep',
       'Sales Executive', 'UX Designer', 'Operations Director',
       'Network Engineer', 'Administrative Assistant',
       'Strategy Consultant', 'Copywriter', 'Account Manager',
       'Director of Marketing', 'Help Desk Analyst',
       'Customer Service Manager', 'Business Intelligence Analyst',
       'Event Coordinator', 'VP of Finance', 'Graphic Designer',
       'UX Researcher', 'Social Media Manager', 'Director of Operations',
       'Senior Data Scientist', 'Junior Accountant',
       'Digital Marketing Manager', 'IT Manager',
       'Customer Service Representative', 'Business Development Manager',
       'Senior Financial Analyst', 'Web Developer', 'Research Director',
       'Technical Support Specialist', 'Creative Director',
       'Senior Software Engineer', 'Human Resources Director',
       'Content Marketing Manager', 'Technical Recruiter',
       'Sales Representative', 'Chief Technology Officer',
       'Junior Designer', 'Financial Advisor', 'Junior Account Manager',
       'Senior Project Manager', 'Principal Scientist',
       'Supply Chain Manager', 'Senior Marketing Manager',
       'Training Specialist', 'Research Scientist',
       'Junior Software Developer', 'Public Relations Manager',
       'Operations Analyst', 'Product Marketing Manager',
       'Senior HR Manager', 'Junior Web Developer',
       'Senior Project Coordinator', 'Chief Data Officer',
       'Digital Content Producer', 'IT Support Specialist',
       'Senior Marketing Analyst', 'Customer Success Manager',
       'Senior Graphic Designer', 'Software Project Manager',
       'Supply Chain Analyst', 'Senior Business Analyst',
       'Junior Marketing Analyst', 'Office Manager', 'Principal Engineer',
       'Junior HR Generalist', 'Senior Product Manager',
       'Junior Operations Analyst', 'Senior HR Generalist',
       'Sales Operations Manager', 'Senior Software Developer',
       'Junior Web Designer', 'Senior Training Specialist',
       'Senior Research Scientist', 'Junior Sales Representative',
       'Junior Marketing Manager', 'Junior Data Analyst',
       'Senior Product Marketing Manager', 'Junior Business Analyst',
       'Senior Sales Manager', 'Junior Marketing Specialist',
       'Junior Project Manager', 'Senior Accountant', 'Director of Sales',
       'Junior Recruiter', 'Senior Business Development Manager',
       'Senior Product Designer', 'Junior Customer Support Specialist',
       'Senior IT Support Specialist', 'Junior Financial Analyst',
       'Senior Operations Manager', 'Director of Human Resources',
       'Junior Software Engineer', 'Senior Sales Representative',
       'Director of Product Management', 'Junior Copywriter',
       'Senior Marketing Coordinator', 'Senior Human Resources Manager',
       'Junior Business Development Associate', 'Senior Account Manager',
       'Senior Researcher', 'Junior HR Coordinator',
       'Director of Finance', 'Junior Marketing Coordinator',
       'Junior Data Scientist', 'Senior Operations Analyst',
       'Senior Human Resources Coordinator', 'Senior UX Designer',
       'Junior Product Manager', 'Senior Marketing Specialist',
       'Senior IT Project Manager', 'Senior Quality Assurance Analyst',
       'Director of Sales and Marketing', 'Senior Account Executive',
       'Director of Business Development', 'Junior Social Media Manager',
       'Senior Human Resources Specialist', 'Senior Data Analyst',
       'Director of Human Capital', 'Junior Advertising Coordinator',
       'Junior UX Designer', 'Senior Marketing Director',
       'Senior IT Consultant', 'Senior Financial Advisor',
       'Junior Business Operations Analyst',
       'Junior Social Media Specialist',
       'Senior Product Development Manager', 'Junior Operations Manager',
       'Senior Software Architect', 'Junior Research Scientist',
       'Senior Financial Manager', 'Senior HR Specialist',
       'Senior Data Engineer', 'Junior Operations Coordinator',
       'Director of HR', 'Senior Operations Coordinator',
       'Junior Financial Advisor', 'Director of Engineering'
]

# Page configuration
st.set_page_config(page_title="Salary Predictor", layout="centered")

# Add a title and description
st.title("💰 Employee Salary Prediction")
st.markdown("""
This application predicts the salary of an employee based on their details.
Fill in the information below to get started!
""")

# Input fields
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", 
                           ["Bachelor's", "Master's", "PhD"])
    job_title = st.selectbox("Job Title", JOB_TITLES)
    experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
    
    submitted = st.form_submit_button("Predict Salary")
    if submitted:
        try:
            # Create input DataFrame
            input_df = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Education Level': [education],
                'Job Title': [job_title],
                'Years of Experience': [experience]
            })
            
            # Preprocess the input data
            input_df['Gender'] = le_gender.transform(input_df['Gender'])
            input_df['Education Level'] = le_education.transform(input_df['Education Level'])
            input_df['Job Title'] = le_job_title.transform(input_df['Job Title'])
            input_df[['Age', 'Years of Experience']] = scaler.transform(input_df[['Age', 'Years of Experience']])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Salary: **${prediction:,.2f}**")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Add some styling for better UI
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 20px;
    }
    .stTitle {
        color: #2C3E50;
    }
    .stMarkdown {
        font-size: 16px;
        color: #34495E;
    }
</style>
""", unsafe_allow_html=True)
