from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server
import pywebio
import pickle
import numpy as np
import webbrowser
from pywebio import STATIC_PATH
import pandas as pd

model = pickle.load(open('heart_disease_random_forest_model.pkl', 'rb'))
app = Flask(__name__)
app._favicon = "favicon.ico"

def edit():
    put_text("You click edit button")
def delete():
    put_text("You click delete button")

def predict():
    """
    Heart Disease Prediction

    Displays the welcome message for the prediction app.

    Returns:
            None
    """
    # put_image('https://images.unsplash.com/photo-1628348070889-cb656235b4eb?ixlib=rb-1.2.1&raw_url=true&q=80&fm=jpg&crop=entropy&cs=tinysrgb&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870')
    # put_image('https://images.unsplash.com/photo-1623134915837-d2fdb4f59035?ixlib=rb-1.2.1&raw_url=true&q=80&fm=jpg&crop=entropy&cs=tinysrgb&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2071')
    # img = open('\\assets\\heart-disease-banner.png', 'rb').read()
    # put_image(img)
    img = open('assets/heart-disease-banner.png', 'rb').read()
    with use_scope('scope1', clear=True):
        put_image(img)
    # put_image("assets\heat-disease-banner.png")
    put_text("")
    # The width ratio of the left and right code blocks is 2:3, which is equivalent to size='2fr 10px 3fr'
    # put_row([put_code('A'), None, put_code('B')], size='40% 10px 60%')
    # put_button("Check for Heart Disease", onclick=lambda: toast("Clicked"), color='success', outline=True)
    # put_buttons(['edit', 'delete'], onclick=[edit, delete])
    # choose_onboarding = actions('Heart Disease Prediction', ['Check for Heart Disease'],
    #                             help_text='')
    # if choose_onboarding == 'Check for Heart Disease':
    #     pass
    # elif choose_onboarding == 'View Dataset':
    #     webbrowser.open('https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset')
    # elif choose_onboarding == 'View Code on Github':
    #     webbrowser.open('https://github.com/mcabanlit/heart-disease')
    # else:
    #     pass
    # put_link('Github', "https://github.com/mcabanlit/heart-disease")
    # welcome()
    from pywebio import start_server

    start = actions('Would you like to start prediction? Still on development mode.', ['Yes', 'No'], help_text='')
    # start = radio("Would you like to start prediction?", options=['Yes', 'No'], required=True)
    if start=='Yes':
        accept = actions('Do you consent the processing of your data?', ['Yes', 'No'],
                         help_text='We will be processing your. \n 1. Age')
        if accept=='Yes':
            age = input("Age of the patient:", type=NUMBER, required=True)
            gender = radio("Gender", options=['Male', 'Female'], required=True)
            if gender == 'Male':
                sex = 1
            else:
                sex = 0
            chest_pain = radio("Chest Pain Type", options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'], required=True)
            if chest_pain == 'Typical Angina':
                cp = 0
            elif chest_pain == 'Atypical Angina':
                cp = 1
            elif chest_pain == 'Non-anginal Pain':
                cp = 3
            else:
                cp = 4

            trestbps = input("Resting Blood Pressure (in mm Hg):", type=NUMBER, required=True)
            chol= input("Cholesterol fetched via BMI sensor (in mg/dl):", type=NUMBER, required=True)
            fasting_blood = input("Fasting blood sugar (in mg/dl)", type=NUMBER, required=True)
            if fasting_blood > 120:
                fbs = 1
            else:
                fbs = 0

            resting_ecg = radio("Resting electrocardiographic results:", options=['Normal', 'ST-T wave normality', 'Left ventricular hypertrophy'], required=True)
            if resting_ecg == 'Normal':
                restecg = 0
            elif resting_ecg == 'ST-T wave normality':
                restecg = 1
            else:
                restecg = 2


            thalach = input("Maximum heart rate achieved:", type=NUMBER, required=True)
            exercise_angina = radio("Exercise induced Angina", options=['Yes', 'No'], required=True)
            if exercise_angina == 'Yes':
                exang = 1
            else:
                exang = 0

            oldpeak = input("Previous peak:", type=FLOAT, required=True)
            slope = input("Slope of Peak Exercise ST segment:", type=NUMBER, required=True)
            ca = input("Number of major vessels:", type=NUMBER, required=True)
            thal = input("Thalium Stress Test result", type=NUMBER, required=True)
            user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            df = pd.DataFrame(user_data, columns=["age", "sex", "cp", "trestbps",
                                       "chol", "fbs", "restecg", "thalach",
                                       "exang", "oldpeak", "slope", "ca", "thal"])
            has_heart_disease = model.predict(df)
            # has_diabetes = model.predict([[age, 1, 0, 125, 212, 0, 1, 169, 0, 1.5, 2, 2, 3]])

            if has_heart_disease == 1:
                verdict = "Has Heart Disease"
            elif has_heart_disease == 0:
                verdict = "No Heart Disease"
            else:
                verdict = "Wowers"

            popup(verdict, [
                put_html('<h4>Details</h4>'),
                put_text('Age: ' + str(age)),
                # 'html: <br/>',
                # put_table([['A', 'B'], ['C', 'D']]),
                put_buttons(['Close Results'], onclick=lambda _: close_popup())
            ])

    predict()

app.add_url_rule('/tool', 'webio_view', webio_view(predict),
                 methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict, port=args.port)
