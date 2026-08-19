# loading in our libraries and cleaned data
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/cleaned_survey.csv')

# preparing data in advance for Random Forest predictor in app
df_clean = df.copy()

# remapping ordinal encoding for 'Age'
age_map = {
    '18-24 years old': 1,
    '25-34 years old': 2,
    '35-44 years old': 3,
    '45-54 years old': 4,
    '55-64 years old': 5,
    '65 years or older': 6
}

# remapping the ordinal encoding for 'EdLevel'
edlevel_map = {
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 1,
    'Some college/university study without earning a degree': 2,
    'Associate degree (A.A., A.S., etc.)': 3,
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 4,
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 5,
    'Professional degree (JD, MD, Ph.D, Ed.D, etc.)': 6,
    'Other (please specify):': 0
}

# remapping the ordinal encoding for 'OrgSize'
# ordinal encoding for orgization size
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

df_clean['Age_encoded'] = df_clean['Age'].map(age_map)
df_clean['EdLevel_encoded'] = df_clean['EdLevel'].map(edlevel_map)
df_clean['OrgSize_encoded'] = df_clean['OrgSize'].map(orgsize_map)

# recreating the one-hot encoded nominal variables
nominal_columns = ['DevType', 'Country', 'Industry', 'Employment', 'ICorPM', 'RemoteWork']
df_encoded = pd.get_dummies(df_clean, columns = nominal_columns, drop_first = True) 

# recreating the multi-label encoding for languages & databases
language_dummies = df_clean['LanguageHaveWorkedWith'].str.get_dummies(sep=';').add_prefix('lang_')
database_dummies = df_clean['DatabaseHaveWorkedWith'].str.get_dummies(sep=';').add_prefix('db_')
df_encoded = pd.concat([df_encoded.drop(columns = ['LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith']), 
                        language_dummies, database_dummies], axis = 1)

# removing the original categorical columns, and max_reasonable_age
columns_to_drop = ['Age', 'EdLevel', 'OrgSize']
if 'max_reasonable_age' in df_encoded.columns:
    columns_to_drop.append('max_reasonable_age')
df_encoded = df_encoded.drop(columns = columns_to_drop)

# setting of X and y for train/test & modeling
X = df_encoded.drop(columns = ['annual_salary_usd', 'ResponseId'])
y = df_encoded['annual_salary_usd']

# using same train/test split used in notebook file to replicate exactly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

# training Random Forest
rf_model = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42,
    n_jobs = 1
)
rf_model.fit(X_train, y_train)

# building our header/title screen
st.title("Developer Salary Predictor")
st.write(
    "Exploring developer salaries and comparing machine learning."
)

# getting tabs set up for what content dashboard will have
tab1, tab2, tab3 = st.tabs([
   'Data Overview',
   'Model Comparison',
   'Salary Predictor'
])

# TAB 1: Data Overview
with tab1:
    st.header("Survey Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric('Respondents', f'{len(df):,}')

    with col2:
        st.metric(
            'Median Salary',
            f'${df['annual_salary_usd'].median():,.0f}'
        )

    st.subheader('Salary Distribution')

    # histogram to display annual_salary_usd
    fig, ax = plt.subplots(figsize = (10, 4))

    ax.hist(df['annual_salary_usd'], bins = 20)

    ax.set_xlabel('Annual Salary (USD)')
    ax.set_ylabel('Number of Respondents')
    ax.set_title('Distribution of Developer Salaries')

    # making the x-axis reflect USD better
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f'${x/1000:.0f}k')
    )

    st.pyplot(fig)

# TAB 2: Model Comparison
with tab2:
    st.header('Model Comparison')
    st.write('Comparing Linear Regression and Random Forest Performance')

    # key metrics
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

    # displays our models' metrics in a comparison table
    comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'R²': [0.533, 0.539],
        'MAE': [30985, 31481],
        'RMSE': [46641, 46337]
    })

    comparison_display = comparison.copy()

    comparison_display['R²'] = comparison_display['R²'].map(lambda x: f'{x:.3f}')
    comparison_display['MAE'] = comparison_display['MAE'].map(lambda x: f'${x:,.0f}')
    comparison_display['RMSE'] = comparison_display['RMSE'].map(lambda x: f'${x:,.0f}')

    st.subheader('Model Performance')

    st.dataframe(
       comparison_display,
       hide_index = True 
    )

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
        dev_type = st.selectbox('Developer Type', sorted(df_clean['DevType'].dropna().unique()))
        country = st.selectbox('Country', sorted(df_clean['Country'].dropna().unique()))
        industry = st.selectbox('Industry', sorted(df_clean['Industry'].dropna().unique()))

    with col2:
        employment = st.selectbox('Employment', sorted(df_clean['Employment'].dropna().unique()))
        manager = st.selectbox('Management Status', sorted(df_clean['ICorPM'].dropna().unique()))
        remote_work = st.selectbox('Remote Work', sorted(df_clean['RemoteWork'].dropna().unique()))

    st.subheader('Technical Experience')

    languages = st.multiselect(
        'Programming Languages',
        sorted(df_clean['LanguageHaveWorkedWith'].dropna().str.split(';').explode().unique())
    )

    databases = st.multiselect(
        'Databases',
        sorted(df_clean['DatabaseHaveWorkedWith'].dropna().str.split(';').explode().unique())
    )

    # fun part! turn those selections into predictions from model
    # creating a dataframe from the user's selections
    input_df = pd.DataFrame({
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

    # recreating the one-hot encoding for categorical variables for user input dataframe
    input_encoded = pd.get_dummies(
        input_df,
        columns = ['DevType', 'Country', 'Industry', 'Employment', 'ICorPM', 'RemoteWork'],
        drop_first = True
    )

    # recreating the multi-label encoding for languages and databases for user input dataframe
    language_input = (
        input_df['LanguageHaveWorkedWith'].str.get_dummies(sep = ';').add_prefix('lang_')
    )

    database_input = (
        input_df['DatabaseHaveWorkedWith'].str.get_dummies(sep = ';').add_prefix('db_')
    )

    user_input_encoded = pd.concat([
        input_encoded.drop(columns = ['LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith']),
        language_input, database_input], axis = 1
    )

    # making sure the prediction row has exactly the same features and column order as Random Forest traning data
    input_encoded = input_encoded.reindex(columns = X.columns, fill_value = 0)

    st.subheader('Predicted Salary')

    if st.button('Predict Salary'):
        prediction = rf_model.predict(input_encoded)[0]

        st.success(f'Estimated Annual Salary: ${prediction:,.0f}')