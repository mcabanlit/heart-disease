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
            age = input("Enter the age of the patient:", type=NUMBER, required=True)
            sex = radio("Gender", options=['Male', 'Female'], required=True)

            user_data = [[age, 1, 1, 140, 221, 0, 1, 164, 1, 0.0, 2, 0, 2]]
            df = pd.DataFrame(user_data, columns=["age", "sex", "cp", "trestbps",
                                       "chol", "fbs", "restecg", "thalach",
                                       "exang", "oldpeak", "slope", "ca", "thal"])
            has_diabetes = model.predict(df)
            # has_diabetes = model.predict([[age, 1, 0, 125, 212, 0, 1, 169, 0, 1.5, 2, 2, 3]])

            if has_diabetes == 1:
                verdict = "Has Heart Disease"
            elif has_diabetes == 0:
                verdict = "No Heart Disease"
            else:
                verdict = "Wowers"

            popup(verdict, [
                put_html('<h3>Popup Content</h3>'),
                'html: <br/>',
                put_table([['A', 'B'], ['C', 'D']]),
                put_buttons(['close_popup()'], onclick=lambda _: close_popup())
            ])

    predict()

app.add_url_rule('/tool', 'webio_view', webio_view(predict),
                 methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict, port=args.port)
