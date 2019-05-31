# Parrot.PY
This program is a prototype meant to explore ways to interact with computer systems and games without using a keyboard and mouse combination. It attempts to achieve this using audio and speech recognition. It also simulates mouse movements with an eye tracker ( in my case a Tobii Eyetracker ) using a seperate program called [Project IRIS](http://iris.xcessity.at/).

It's name is inspired by the way parrots and parakeets communicate, using chirps, clicks and sometimes speech. 

# Software requirements
* Windows version 7 and up.
* Project IRIS ( used for turning the eyetracker into a mouse cursor )
* Python 3.6 (32 bit)

Python packages such as
* numpy
* pandas 
* matplotlib *( for the graphing of test results )*
* scikit-learn *( for the machine learning bits )*
* pyaudio *( audio recording and playing )*
* python_speech_features *( for audio manipulation, specifically the MFCC algorithm )*
* pyautogui *( for mouse and keyboard simulation )*
* dragonfly *( For speech recognition purposes )*
* pythoncom *( for listening to speech recognition commands )*

# Hardware requirements
* A decent computer to run machine learning training on
* An audio recording device, a cheap multi directional microphone will suffice
* Mouse and keyboard, for configuration purposes
* Eye tracker device ( if you want to use this program without a mouse )

# Installation

Follow the instructions in the [Installation guide for this project](docs/INSTALLATION.md)

# Configuration

This repository contains a settings.py file for recording, training and analytical purposes. 

Step 1 - Recording data

In order to train a model, you need to record sounds first. You can do this by running settings.py and pressing the [R] key, this will lead you through the steps neccesary for recording.
All the recorded files are saved on your computer in seperate folders, designated with the names of your choosing in the recording process. Recording can be paused and exited using the space and escape key respectively.

It is recommended to record around 1000 samples per sound, this is what I found to be the best middle ground for good detection.

Make sure to also record background noise under the folder 'silence'. The models require this directory to be available.

Step 2 - Training the model

Training the model involves going through the [L] menu when running settings.py . Here you will choose the model name and which sounds you want to recognize.
Going through this menu will automatically generate a trained model, with accuracy predictions for the specific sounds.

Step 3 - Analyzing performance of the model

You can analyze the model performance when you go through the [A] menu when running settings.py. This will generate graphs which can be used to see how well a certain sound is being detected.

There are two ways to analyze performance. 
1. Generating a graph of the detected sounds using the replay files ( located in data/replays ).
2. Choosing a model and recording a brief segment of sound files to see if the model holds up to your expectations. 
This option is prefered when you have just generated a model and want to see how well it detects sounds outside of the sample set of the prerecorded sounds.

With this option, you can also choose to use the previously recorded files, in order to see how other models match up against the same set of sounds.

Step 4 - Tweaking modes

As it stands right now, there is no easy way to do the configuration of the given interaction modes. You will have to edit the code yourself.
The modes can be found in the lib/modes/ directory.

Most of this tweaking will include changing the pattern detector configuration in the top of the files by changing the sounds and their percentage and intensity thresholds.

There keys should not be changed, however, inside its contents, you can tweak the following elements:
Sound - This is the name of the sound you wish to use when this key should be activated
Strategy - You can choose between.. TODO

# Running the program

You can run the program by executing `py play.py`. This process might crash if you haven't properly installed certain packages or configurated your models.

### Todos

 - Add examples of how to use the program
 - Seperate out the eyetracker and speech recognition parts, in order to allow the users not to have to install and configure all the different parts in order to get set up properly. 