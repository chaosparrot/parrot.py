# Parrot.PY
This program is a prototype meant to explore ways to interact with computer systems and games without using a keyboard and mouse combination. It attempts to achieve this using audio and speech recognition. It also simulates mouse movements with an eye tracker ( in my case a Tobii Eyetracker ) using a seperate program called [Project IRIS](http://iris.xcessity.at/).

It's name is inspired by the way parrots and parakeets communicate, using chirps, clicks and sometimes speech. 

# Software requirements
* Windows version 7 and up, MacOS or Linux (X11)
* Python 3.8 (64 bit recommended)
* Project IRIS ( OPTIONAL - Used for turning the eyetracker into a mouse cursor )

Python packages such as
* numpy
* pandas 
* matplotlib *( for the graphing of test results )*
* scikit-learn *( for the machine learning bits )*
* pyaudio *( audio recording and playing )*
* python_speech_features *( for audio manipulation, specifically the MFCC algorithm )*
* pyautogui *( for mouse and keyboard simulation )*
* pydirectinput *( for mouse and keyboard simulation, but with DirectX keycodes instead )*
* dragonfly2 *( For speech recognition purposes )*
* pythoncom *( for listening to speech recognition commands )*
* pytorch *( Used for improved neural nets )*
* audiomentations *( Used in training for generating more samples to train on ) 

# Hardware requirements
* A decent computer to run machine learning training on
* An audio recording device, a cheap multi directional microphone will suffice
* Mouse and keyboard, for configuration purposes
* Eye tracker device ( if you want to use this program without a mouse )

# Installation

Follow the instructions in the [Installation guide for this project](docs/INSTALLATION.md)

# Configuration

![Settings example](docs/media/settings-main.png)

This repository contains a settings.py file for recording, training and analytical purposes. 
This will allow you to tweak and improve the recognition of sounds and activation accuracy and speed of keys.

In order to map a sound to a key, you will have to go through a certain process:
* Step 1 - [Recording the sound](docs/RECORDING.md) or [Converting the sound](docs/CONVERTING.md)
* Step 2 - [Training a model on the sound for recognition](docs/TRAINING.md)
* Step 3 - [Analyse whether or not the sound is properly recognized](docs/ANALYSING.md)
* Step 4 - [Connecting sounds to keyboard and mouse inputs](docs/TUTORIAL_MODE.md)

# Running the program

You can run the program by executing `py play.py`. This process might crash if you haven't properly installed certain packages or configurated your models.

# Errors running the program

Sometimes if windows speech rec isnt started before the program, there appears to be a weird caching bug that shows the following error:
Exception while initializing sapi5 engine: module 'win32com.gen_py' has no attribute 'CLSIDToClassMap'
You can fix that by starting windows speech rec, going to your C:\Users\YOURUSER\AppData\Local\Temp directory and rename gen_py to gen_py2 or something. This will rebuild the cache for gen_py the next time the program is run.

# Using Parrot.PY with an On Screen Keyboard in Windows

The OSK program in Windows is a special program that doesn't trust the input that this python program is outputting. So if you try clicking on the OSK buttons, it doesn't respond.
A way to mitigate that is by running your command line program as an administrator. You can do this by searching for your command line program in the start menu, and right clicking on the option 'Run as administrator'.

# Other related stuff

For instructions on upgrading and the release notes, look on ![this page](docs/RELEASE_NOTES.md).

I have a seperated doc page filled with research, notes, acknowledgements and showcases over on ![this page](docs/NOTES_AND_ACKNOWLEDGEMENTS.md).
I will expand this document once I learn more during the development. In the meantime, you can use that knowledge to your own advantage.

If you desire to use Parrot.PY inside of Talon, take a look at this page explaining the [integration with Talon Voice](docs/TALON_VOICE.md).
For examples and explanation of how to interact with a running Parrot instance from your own programs, read up on [interfacing with other programs](docs/INTERFACING.md).