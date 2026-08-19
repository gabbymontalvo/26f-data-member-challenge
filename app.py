# loading libaries for streamlit to build the dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# used to recreate the same train/test split and train the Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# applying a consistent theme to the app using CSS
st.markdown("""
<style>
    /* Main app background */
    .stApp{
        background-color: #0B1220;
        color: #F4F6F8;
    }

    /* Main text */
    p, label, .stMarkdown {
        color: #F4F6F8;
    }

    /* Metric cards */
    [data-testid = "stMetric"] {
        background-color: #111C2E;
        border: 1px solid #24344D;
        border-radius: 12px;
        padding: 18px;
    }

    /* Metric labels */
    [data-testid = "stMetricLabel"] {
        color: #AAB5C5;
    }

    /* Metric values */
    [data-testid = "stMetricValue"] {
        color: #F4F6F8;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4A90E2;
        color: #F4F6F8;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #76B5F5;
        color: #0B1220;
    }
</style>
""", unsafe_allow_html = True)

# loading in the cleaned survey data for the dashboard
df = pd.read_csv('data/cleaned_survey.csv')

# loading the encoded dataset used to train the Random Forest
# this will keep the app's training data consistent with the modeling notebook
df_encoded = pd.read_csv('data/encoded_survey.csv')

# these maps will convert the original categorical survey responses into numerical values for the model
# need to use these for when the user inputs their responses in the app
age_map = {
    '18-24 years old': 1,
    '25-34 years old': 2,
    '35-44 years old': 3,
    '45-54 years old': 4,
    '55-64 years old': 5,
    '65 years or older': 6
}

edlevel_map = {
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 1,
    'Some college/university study without earning a degree': 2,
    'Associate degree (A.A., A.S., etc.)': 3,
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 4,
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 5,
    'Professional degree (JD, MD, Ph.D, Ed.D, etc.)': 6,
    'Other (please specify):': 0
}

orgsize_map = {
    'Just me - I am a freelancer, sole proprietor, etc.': 1,
    'Less than 20 employees': 2,
    '20 to 99 employees': 3,
    '100 to 499 employees': 4,
    '500 to 999 employees': 5,
    '1,000 to 4,999 employees': 6,
    '5,000 to 9,999 employees': 7,
    '10,000 or more employees': 8
}

# separate the target (salary) from the model features
# ResponseId is also removed because it identifies the response rather than giving useful information
X = df_encoded.drop(columns = ['annual_salary_usd', 'ResponseId'])
y = df_encoded['annual_salary_usd']

# using the same 80/20 split and random state as the modeling notebook so model is evaluated and reproduced correctly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

# training Random Forest selected during model comparison
rf_model = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42,
    n_jobs = 1
)
rf_model.fit(X_train, y_train)

# main title and short description for the application
# using markdown for more detailing
st.markdown(
    """
    <div class = "hero">
        <div class = "eyebrown" > DATA EXPLORATION • MACHINE LEARNING</div>
        <h1>Developer Salary Lab</h1>
        <p>
                Explore salary patterns, compare predictive models, 
                and estimate compensation based on developer experience.
        </p>
    </div>
""", unsafe_allow_html = True)

# diving the dashboard into three setcions:
# dataset exploration/overview, model comparisons, and interactive prediction
tab1, tab2, tab3 = st.tabs([
   '01 Data',
   '02 Models',
   '03 Predict'
])

# TAB 1: Data Overview
with tab1:
    st.header("Survey Overview")

    # displays two quick smmary statistics at the top of the tab
    col1, col2 = st.columns(2)

    with col1:
        st.metric('Respondents', f'{len(df):,}')

    with col2:
        st.metric(
            'Median Salary',
            f'${df['annual_salary_usd'].median():,.0f}'
        )

    st.subheader('Salary Distribution')

    # visualizes how salaries are distributed across survey respondents
    fig, ax = plt.subplots(figsize = (10, 4))

    ax.hist(df['annual_salary_usd'], bins = 20)

    ax.set_xlabel('Annual Salary (USD)')
    ax.set_ylabel('Number of Respondents')
    ax.set_title('Distribution of Developer Salaries')

    # formatting the x-axis as dollar amounts in thousands
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f'${x/1000:.0f}k')
    )

    st.pyplot(fig)

# TAB 2: Model Comparison
with tab2:
    st.header('Model Comparison')
    st.write('Comparing Linear Regression and Random Forest Performance')

    # highlighting the strongest result for each evaluation metric
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            'Best R²',
            '0.539',
            'Random Forest'
        )
    with col2:
        st.metric(
            'Best MAE',
            '$30,985',
            'Linear Regression'
        )
    with col3:
        st.metric(
            'Best RMSE',
            '$46,337',
            'Random Forest'
        )

    # stores the final metrics from both models in one table to compare performance
    comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'R²': [0.533, 0.539],
        'MAE': [30985, 31481],
        'RMSE': [46641, 46337]
    })

    # creating a formatted copy for displaying dollar values cleanly
    comparison_display = comparison.copy()

    comparison_display['R²'] = comparison_display['R²'].map(lambda x: f'{x:.3f}')
    comparison_display['MAE'] = comparison_display['MAE'].map(lambda x: f'${x:,.0f}')
    comparison_display['RMSE'] = comparison_display['RMSE'].map(lambda x: f'${x:,.0f}')

    st.subheader('Model Performance')

    st.dataframe(
       comparison_display,
       hide_index = True 
    )

    # summarizes the main takeaway from the comparison
    st.info(
        'Random Forest achieved slightly higher R² and slightly lower ' \
        'RMSE, while Linear Regression had a lower MAE. Overall, ' \
        'the models performed similarly.'
    )

# TAB 3: Salary Predictor
with tab3:
    st.header('Salary Predictor')
    st.write('Enter developer characteristics to estimate annual salary!')
    st.subheader('Developer Information')

    # collecting the characteristics that will be used as the model inputs
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox('Age', list(age_map.keys()))
        education = st.selectbox('Education Level', list(edlevel_map.keys()))
        years_code = st.number_input('Years of Coding Experience', min_value = 0, max_value = 60, value = 5)

    with col2:
        work_exp = st.number_input('Years of Professional Experience', min_value = 0, max_value = 60, value = 3)
        org_size = st.selectbox('Organization Size', list(orgsize_map.keys()))

    st.subheader('Work Information')

    col1, col2 = st.columns(2)

    with col1:
        dev_type = st.selectbox('Developer Type', sorted(df['DevType'].dropna().unique()))
        country = st.selectbox('Country', sorted(df['Country'].dropna().unique()))
        industry = st.selectbox('Industry', sorted(df['Industry'].dropna().unique()))

    with col2:
        employment = st.selectbox('Employment', sorted(df['Employment'].dropna().unique()))
        manager = st.selectbox('Management Status', sorted(df['ICorPM'].dropna().unique()))
        remote_work = st.selectbox('Remote Work', sorted(df['RemoteWork'].dropna().unique()))

    st.subheader('Technical Experience')

    languages = st.multiselect(
        'Programming Languages',
        sorted(df['LanguageHaveWorkedWith'].dropna().str.split(';').explode().unique())
    )

    databases = st.multiselect(
        'Databases',
        sorted(df['DatabaseHaveWorkedWith'].dropna().str.split(';').explode().unique())
    )

    # fun part! turn those selections into predictions from model
    # creating a dataframe from the user's selections, column names must match original survey data
    # makes it easier to apply the same transformations used during model training
    user_input = pd.DataFrame({
        'WorkExp': [work_exp],
        'YearsCode': [years_code],
        'Age_encoded': [age_map[age]],
        'EdLevel_encoded': [edlevel_map[education]],
        'OrgSize_encoded': [orgsize_map[org_size]],
        'DevType': [dev_type],
        'Country': [country],
        'Industry': [industry],
        'Employment': [employment],
        'ICorPM': [manager],
        'RemoteWork': [remote_work],
        'LanguageHaveWorkedWith': [';'.join(languages)],
        'DatabaseHaveWorkedWith': [';'.join(databases)]
    })

    # convert the categorical selections into one-hot encoded columns
    # matches the approach used when prepping the training data
    user_categorical = pd.get_dummies(
        user_input,
        columns = ['DevType', 'Country', 'Industry', 'Employment', 'ICorPM', 'RemoteWork'],
        drop_first = True
    )

    # converts the user's multiple language selections into separate binary columns
    # 1 indicates that a language was selected
    language_input = (
        user_input['LanguageHaveWorkedWith'].str.get_dummies(sep = ';').add_prefix('lang_')
    )

    # the same multi label encoding for database selections (like languages)
    database_input = (
        user_input['DatabaseHaveWorkedWith'].str.get_dummies(sep = ';').add_prefix('db_')
    )

    # combining all the encoded inputs into one row
    user_input_encoded = pd.concat([
        user_categorical.drop(columns = ['LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith']),
        language_input, database_input], axis = 1
    )

    # making sure the prediction row has exactly the same features and column order as Random Forest traning data
    # any feature the user did not select is filled with 0
    user_input_encoded = user_input_encoded.reindex(columns = X.columns, fill_value = 0)

    st.subheader('Predicted Salary')

    # generates a prediction when the user clicks the button
    if st.button('Predict Salary'):
        prediction = rf_model.predict(user_input_encoded)[0]

        st.success(f'Estimated Annual Salary: ${prediction:,.0f}')