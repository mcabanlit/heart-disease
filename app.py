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

model = pickle.load(open('heart_disease_random_forest_model.pkl', 'rb'))
app = Flask(__name__)
app._favicon = "favicon.ico"

def edit():
    put_text("You click edit button")
def delete():
    put_text("You click delete button")

def welcome():
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
    start = radio("Would you like to start prediction?", options=['Yes', 'No'], required=True)
    if start=='Yes':
        accept = radio("Do you consent the processing of your data?", options=['Yes', 'No'], required=True)
        if accept=='Yes':
            age = input("Enter the age of the patient:", type=NUMBER, required=True)
            sex = radio("Gender", options=['Male', 'Female'], required=True)


app.add_url_rule('/tool', 'webio_view', webio_view(welcome),
                 methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(welcome, port=args.port)
