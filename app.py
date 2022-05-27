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
from pywebio import STATIC_PATH
import pandas as pd

model = pickle.load(open('heart_disease_random_forest_model.pkl', 'rb'))
app = Flask(__name__)
app._favicon = "favicon.ico"

my_buttons = [{'label': 'Submit', 'value': 'submit', 'type': 'submit', 'color': 'info'},
              {'label': 'Reset', 'value': 'reset', 'type': 'reset', 'color': 'secondary'}]


def predict():
    """
    Heart Disease Prediction

    Displays the welcome message for the prediction app.

    Returns:
            None
    """

    # For cleaning up the data, you may refer to:
    # https: // towardsdatascience.com / exploratory - data - analysis - on - heart - disease - uci - data - set - ae129e47b323

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
                         placeholder="246", help_text="Requires the serum cholesterol in mg/dl. "
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
                                required=True, value="Normal", help_text="The resting electrocardiographic results"
                                                                         "of the individual which will then be "
                                                                         "converted into numerical values "
                                                                         "for the model.")
            restecg = convert_resting_ecg(resting_ecg)

            thalach = input("Max heart rate achieved:", type=NUMBER, required=True, placeholder="150",
                            help_text="The maximum heart rate achieved by an individual. The dataset contains"
                                      "values ranging from 71 up unto 202")

            exercise_angina = radio("Exercise induced Angina", options=['Yes', 'No'], required=True, value="Yes",
                                    help_text="Angina tends to appear during physical activity, emotional stress, "
                                              "or exposure to cold temperatures, or after big meals. "
                                              "Symptoms of angina include: pressure, aching, or burning in the "
                                              "middle of the chest. pressure, aching, or burning in the neck, jaw, "
                                              "and shoulders (usually the left shoulder) and even down the arm."
                                              "- Harvard Health")
            exang = convert_exercise_induced_angina(exercise_angina)

            oldpeak = input("ST depression induced by exercise relative to rest:", type=FLOAT,
                            required=True, placeholder="1.1",
                            help_text="Exercise induced ST segment depression is considered a reliable ECG finding for "
                                      "the diagnosis of obstructive coronary atherosclerosis. The dataset contains"
                                      "values up until 6.2.")
            slope_of_peak = radio("Peak Exercise ST segment:", options=['Upsloping', 'Flat', 'Downsloping'],
                                  required=True, value="Flat", help_text="Slope of the peak exercise ST segment.")
            slope = convert_slope_peak(slope_of_peak)

            ca = input("Number of major vessels:", type=NUMBER, required=True, placeholder="1",
                       help_text="The number of major vessels (0-3).")

            thalassemia = radio("Blood disorder (thalassemia)", options=['Normal', 'Fixed Defect', 'Reversible Defect'],
                                value="Normal", required=True,
                                help_text="Presence of a blood disorder called thalassemia. If the value is Normal "
                                          "then the individual has normal blood flow, if fixed defect then there "
                                          "is no blood flow in some part of the heart and if it is reversible defect "
                                          "then it means that a blood flow can be observed but not normal.")
            thal = convert_thalassemia(thalassemia)


            user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, 1.1, slope, ca, thal]]
            # user_data = [[age, sex, cp, trestbps, chol, fbs, restecg, 169, 0, 1.5, 2, 2, 3]]
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
        # webbrowser.open('https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset')

        popup("Opening an external link ⚠️", [
            put_text('The heart disease prediction app would like to open the external link below. Please click '
                     'the hyperlink below if you wish to continue, if not you may close this popup window.'),
            put_html('<a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Visit Kaggle 📊</a>'),
            put_text(' '),
            put_buttons(['Close'], onclick=lambda _: close_popup())
        ])

    elif welcome['action'] == 'browse_code':
        # webbrowser.open_new_tab('https://github.com/mcabanlit/heart-disease')

        popup("Opening an external link ⚠️", [
            put_text('The heart disease prediction app would like to open the external link below. Please click '
                     'the hyperlink below if you wish to continue, if not you may close this popup window.'),
            put_html('<a href="https://github.com/mcabanlit/heart-disease">Visit GitHub 🗂️</a>'),
            put_text(' '),
            put_buttons(['Close'], onclick=lambda _: close_popup())
        ])

    predict()


def convert_thalassemia(thalassemia):
    if thalassemia == 'Normal':
        thal = 1
    elif thalassemia == 'Fixed Defect':
        thal = 2
    elif thalassemia == 'Reversible Defect':
        thal = 3
    else:
        thal = 0
    return thal


def convert_slope_peak(slope_of_peak):
    if slope_of_peak == 'Upsloping':
        slope = 0
    elif slope_of_peak == 'Flat':
        slope = 1
    elif slope_of_peak == 'Downsloping':
        slope = 2
    return slope


def convert_exercise_induced_angina(exercise_angina):
    if exercise_angina == 'Yes':
        exang = 1
    else:
        exang = 0
    return exang


def convert_resting_ecg(resting_ecg):
    if resting_ecg == 'Normal':
        restecg = 0
    elif resting_ecg == 'ST-T wave normality':
        restecg = 1
    else:
        restecg = 2
    return restecg


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
