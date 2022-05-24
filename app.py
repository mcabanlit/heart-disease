from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server

import pickle
import numpy as np
model = pickle.load(open('heart_disease_random_forest_model.pkl', 'rb'))
app = Flask(__name__)

def welcome():
    choose_onboarding = actions('Welcome to Car Booking App 🚖', ['Login', 'Signup'],
                                help_text='Choose one of the options to proceed.')

    if choose_onboarding == 'Login':
        login()
    else:
        signup()
        
        
app.add_url_rule('/tool', 'webio_view', webio_view(welcome),
            methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(welcome, port=args.port)