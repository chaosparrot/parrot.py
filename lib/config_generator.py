import os
from config.config import *

if (DEFAULT_CLF_FILE == ""):
    print("---- CONFIGURATION ERROR ----")
    available_models = []
    for fileindex, file in enumerate(os.listdir( CLASSIFIER_FOLDER )):
        if ( file.endswith(".pkl") ):
            available_models.append( file )

    print("You haven't set up a model file (DEFAULT_CLF_FILE) in your data/code/config.py file yet!") 
    if (len(available_models) > 0):
        print("Make sure to enter the filename ( without the file extension ) of one of the models in " + CLASSIFIER_FOLDER + " to load in the model")
        print("It seems you have already trained models, try setting one of the following options:")
        for available_model in available_models:
            print(available_model.replace(".pkl", ""))
    else:
        print("You also haven't trained a model yet. Make sure to install a model or train it using the setup menu. Instructions can be found in the README.md file.")
    print("-----------------------------")    
    exit()

if (STARTING_MODE == ""):
    print("---- CONFIGURATION ERROR ----")
    available_files = []
    for fileindex, file in enumerate(os.listdir( 'data/code' )):
        if file != "config.py" and file.endswith(".py"):
            available_files.append(file)
    
    print("You haven't set up a starting mode in your data/code/config.py file yet!")
    if (len(available_files) > 0):
        print("It seems you have files set up in your data/code folder, try setting the STARTING_MODE to one of the following:")
        for available_file in available_files:
            print(available_file.replace(".py", ""))
    else:
        print("You do not have mode files inside of data/code.")
        print("Copy a mode file or follow the instructions in docs/TUTORIAL_MODE.md to get set up.")
    print("-----------------------------")
    exit()
