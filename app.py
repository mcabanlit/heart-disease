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

my_buttons = [{'label': 'Submit', 'value': 'submit', 'type': 'submit', 'color': 'info'},
              {'label': 'Reset', 'value': 'reset', 'type': 'reset', 'color': 'secondary'}]


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
    put_text("")

    welcome = input_group('What would you like to do?', [
        # input('username', type=TEXT, name='username', required=True),
        # input('password', type=PASSWORD, name='password', required=True),
        actions('This web app uses Random Forest classifier on more than a thousand datasets on heart-disease in order '
                'to try and predict if a user has heart disease. Please choose one of the options below to proceed.', [
                    {'label': 'Check for Heart Disease', 'value': 'make_prediction', 'color': 'info'},
                    {'label': 'View Dataset', 'value': 'view_dataset', 'color': 'secondary'},
                    {'label': 'Browse Code', 'value': 'browse_code', 'color': 'dark'},
                ], name='action',
                help_text='This model uses factors such as age, sex, chest pain, blood pressure, serum '
                          'cholesterol, fasting blood sugar, resting ecg, maximum heart rate, exercise'
                          'induced angina and number of major vessels among others.'),
    ])

    # start = actions('Would you like to start prediction?', ['Yes', 'No'], help_text='')
    # start = radio("Would you like to start prediction?", options=['Yes', 'No'], required=True)
    if welcome['action'] == 'make_prediction':
        accept = actions('Do you consent the processing of your data?', [
            # 'Yes', 'No'
            {'label': 'Yes, I consent.', 'value': 'i_consent', 'color': 'info'},
            {'label': 'No, I do not consent.', 'value': 'predict', 'color': 'dark'},
        ], help_text='We will not be storing any of the values that you have entered after the prediction. '
                     'Please also note that the results of this forecast are not final and is only a product of '
                     'a model that was trained using a publicly available dataset.')

        if accept == 'i_consent':
            with use_scope('scope1', clear=True):
                put_text("")
            # Data description can be found in 7
            age = input("Age of the patient:", type=NUMBER, required=True,
                        placeholder='56', help_text='Age of the individual in years.')

            gender = radio("Gender", options=['Male', 'Female'], value='Male', required=True,
                           help_text="Gender of the individual.")
            sex = convert_gender(gender)

            chest_pain = radio("Chest Pain Type",
                               options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
                               required=True, help_text="Type of chest-pain experienced by the individual.",
                               value="Typical Angina")
            cp = convert_chest_pain(chest_pain)

            trestbps = input("Resting Blood Pressure (in mmHg):", type=NUMBER, required=True, placeholder="131",
                             help_text="The resting blood pressure value of an individual in mmHg. The "
                                       "dataset contains blood pressure ranging from 94mmHg - 200mmHg")

            chol = input("Cholesterol fetched via BMI sensor (in mg/dl):", type=NUMBER, required=True,
                         placeholder="246", help_text="Displays the serum cholesterol in mg/dl. "
                                                          "The dataset contains cholesterol values ranging from"
                                                          "126mg/dl - 564mg/dl.")

            fasting_blood = input("Fasting blood sugar (in mg/dl)", type=NUMBER, required=True, placeholder="120",
                                  help_text="The fasting blood sugar of an individual. The model compares the fasting "
                                            "blood sugar value of an individual with 120mg/dl."
                                            " So the fasting blood sugar will be divided into two categories, "
                                            "greater than the threshold or less than or equal to the threshold.")
            fbs = convert_fasting_blood_sugar(fasting_blood)

            resting_ecg = radio("Resting electrocardiographic results:",
                                options=['Normal', 'ST-T wave normality', 'Left ventricular hypertrophy'],
                                required=True, value="Normal")
            if resting_ecg == 'Normal':
                restecg = 0
            elif resting_ecg == 'ST-T wave normality':
                restecg = 1
            else:
                restecg = 2

            thalach = input("Maximum heart rate achieved:", type=NUMBER, required=True, placeholder="71-202")
            exercise_angina = radio("Exercise induced Angina", options=['Yes', 'No'], required=True, value="Yes")
            if exercise_angina == 'Yes':
                exang = 1
            else:
                exang = 0

            oldpeak = input("ST depression induced by exercise relative to rest (Previous peak):", type=FLOAT,
                            required=True, placeholder="0.0 - 6.2")
            slope_of_peak = radio("Slope of Peak Exercise ST segment:", options=['Upsloping', 'Flat', 'Downsloping'],
                          required=True, value="Flat")
            if slope_of_peak == 'Upsloping':
                slope = 0
            elif slope_of_peak == 'Flat':
                slope = 1
            elif slope_of_peak == 'Downsloping':
                slope = 2

            ca = input("Number of major vessels:", type=NUMBER, required=True, placeholder="0 - 4")
            thalassemia = radio("Blood disorder (thalassemia)", options=['Normal', 'Fixed Defect', 'Reversible Defect'],
                                value="Normal", required=True)
            if thalassemia == 'Normal':
                thal = 1
            elif thalassemia == 'Fixed Defect':
                thal = 2
            elif thalassemia == 'Reversible Defect':
                thal = 3
            else:
                exang = 0

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
                put_html('<h4>The result was based on the following values:</h4>'),
                # put_text('Age: ' + str(age)),
                # 'html: <br/>',
                put_table([['Predictor', 'Value'],
                           ['Age', str(age) + " years old"],
                           ['Sex', gender],
                           ['Chest Pain Type', chest_pain],
                           ['Resting Blood Pressure', str(trestbps) + " mmHg"],
                           ['Cholesterol', str(chol) + " mg/dl"],
                           ['Fasting Blood Sugar', str(fasting_blood) + ' mg/dl'],
                           ['Resting ECG', str(resting_ecg)],
                           ['Maximum Heart Rate', str(thalach)],
                           ['Exercise induced Angina', exercise_angina],
                           ['Previous Peak', str(oldpeak)],
                           ['Slope of Peak', slope_of_peak],
                           ['No of Major Vessels', str(ca)],
                           ['Thalassemia', thalassemia]]),
                put_buttons(['Close Results'], onclick=lambda _: close_popup())
            ])
    elif welcome['action'] == 'view_dataset':
        webbrowser.open('https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset')

        popup("Ok ra ka dol?", [
            put_html('<a href="https://www.w3schools.com/">Visit W3Schools.com!</a>'),
            put_buttons(['Close Results'], onclick=lambda _: close_popup())
        ])

    elif welcome['action'] == 'browse_code':
        webbrowser.open_new_tab('https://github.com/mcabanlit/heart-disease')

    predict()


def convert_fasting_blood_sugar(fasting_blood):
    if fasting_blood > 120:
        fbs = 1
    else:
        fbs = 0
    return fbs


def convert_chest_pain(chest_pain):
    if chest_pain == 'Typical Angina':
        cp = 0
    elif chest_pain == 'Atypical Angina':
        cp = 1
    elif chest_pain == 'Non-anginal Pain':
        cp = 3
    else:
        cp = 4

    return cp


def convert_gender(gender):
    if gender == 'Male':
        sex = 1
    else:
        sex = 0
    return sex


app.add_url_rule('/tool', 'webio_view', webio_view(predict),
                 methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict, port=args.port)
