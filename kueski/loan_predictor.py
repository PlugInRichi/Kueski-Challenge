from joblib import load
import numpy as np
import pandas as pd

model = load('model_risk.joblib')
NAME_INCOME_TYPES = ['Commercial associate',
                     'Pensioner',
                     'State servant',
                     'Student',
                     'Working']
INCOME_TYPES = dict(zip(NAME_INCOME_TYPES, range(len(NAME_INCOME_TYPES))))
ZEROS = [0 for i in range(len(NAME_INCOME_TYPES))]

# Simulate the data base
df = pd.read_csv('dataset_credit_risk.csv')
df["loan_date"] = pd.to_datetime(df.loan_date)
BOOL_FLAG = {'Y': 1, 'N': 0}



def get_prediction(user_id):
    """Compute credit risk to one user"""
    user_data = get_latest_user_data(user_id)
    income_type_1hot = ZEROS.copy()
    income_type_1hot.insert(INCOME_TYPES[user_data[-1]], 1)
    data = user_data[:-1] + income_type_1hot

    return model.predict(np.array(data, ndmin=2))

def get_latest_user_data(user_id):
    """Brings up the latest data for a specific user"""

    user_information = df[df['id'] == user_id].sort_values(by=["id", "loan_date"])
    last_user_information = user_information.iloc[-1]

    loans_previous = user_information['loan_amount'].iloc[:-1]

    today = pd.to_datetime('today').normalize()
    birthday = pd.to_datetime(last_user_information['birthday'], errors='coerce')
    job_start_date = pd.to_datetime(last_user_information['job_start_date'], errors='coerce')

    avg_amount_loans_previous = loans_previous.mean()
    nb_previous_loans = loans_previous.count()
    age = (today - birthday).days // 365
    years_on_the_job = (today - job_start_date).days // 365
    flag_own_car = BOOL_FLAG[last_user_information['flag_own_car']]
    flag_own_realty = BOOL_FLAG[last_user_information['flag_own_realty']]
    cnt_fam_members = last_user_information['cnt_fam_members']
    amt_income_total = last_user_information['amt_income_total']
    name_income_type = last_user_information['name_income_type']

    return [avg_amount_loans_previous,
            nb_previous_loans,
            age,
            years_on_the_job,
            flag_own_car,
            flag_own_realty,
            cnt_fam_members,
            amt_income_total,
            name_income_type]
