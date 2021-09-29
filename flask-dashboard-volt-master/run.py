# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from __future__ import division, print_function
from json import dump
import sys
import os
import glob
import re
import numpy as np
import cv2

from flask_migrate import Migrate
from sys import exit
from decouple import config


from config import config_dict
from app import create_app, db
from flask import render_template, redirect, url_for, request,Flask
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template,flash
from werkzeug.utils import secure_filename


from flask_migrate import Migrate
from os import environ
from sys import exit
from decouple import config
import logging

from config import config_dict
from app import create_app, db

MODEL_PATH = 'models/transfer_learning_mobilenetupdated_model.h5'
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = './app/base/static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


app=Flask(__name__  ,static_url_path="", static_folder="uploads")
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# WARNING: Don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True, cast=bool)

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:
    
    # Load the configuration using the default values 
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app( app_config ) 
Migrate(app, db)

if DEBUG:
    app.logger.info('DEBUG       = ' + str(DEBUG)      )
    app.logger.info('Environment = ' + get_config_mode )
    app.logger.info('DBMS        = ' + app_config.SQLALCHEMY_DATABASE_URI )

if __name__ == "__main__":
    app.run()




@app.route("/predection",methods=['GET', 'POST'])
def predection() :
    if request.method == 'POST':
        
       
        f = request.files['file']
        if f:
        #basepath = os.path.dirname(__file__)
            file_path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(file_path)
      
            image = cv2.imread(file_path) # read file 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
            image = cv2.resize(image,(224,224))
            image = np.array(image) / 255
            image = np.expand_dims(image, axis=0)

            print(image)

        
   
        # Make prediction
            preds = model.predict(image)
            print(preds)
    


            probability = preds[0]
            if probability[0] >= 0.5:
                skin_cancer_pred = str('%.2f' % (probability[0]*100) + '% Bengin') 
            else:
                skin_cancer_pred = str('%.2f' % ((1-probability[0])*100) + '% Malignant')
            print(skin_cancer_pred)
            return render_template('predections.html',pred=preds,image=f.filename,skin_cancer_pred=skin_cancer_pred)
        
        return render_template('predections.html',preds=0)


    
   
@app.route('/dash', methods=['GET', 'POST'])
def dash():

       
    return render_template('dashboard.html')


