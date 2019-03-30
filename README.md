# Parrot.PY
This program is a prototype meant to explore ways to interact with computer systems and games without using a keyboard and mouse combination. It attempts to achieve this using audio and speech recognition. It also simulates mouse movements with an eye tracker ( in my case a Tobii Eyetracker ) using a seperate program called [Project IRIS](http://iris.xcessity.at/).

It's name is inspired by the way parrots and parakeets communicate, using chirps, clicks and sometimes speech. 

# Software requirements
* Windows version 7 and up.
* Project IRIS ( used for turning the eyetracker into a mouse cursor )
* Python 3.6 (32 bit)
* nircmd.exe ( used for toggling system volume )

Python packages such as
* numpy
* pandas 
* matplotlib *( for the graphing of test results )*
* scikit-learn *( for the machine learning bits )*
* pyaudio *( audio recording and playing )*
* audioop *( audio manipulation )*
* python_speech_features *( for audio manipulation, specifically the MFCC algorithm )*
* pyautogui *( for mouse and keyboard simulation )*
* dragonfly *( installation for Python 3 is rather iffy, will add a link to the version I used to get up and running later )*
* pythoncom *( for listening to speech recognition commands )*
* msvcrt *( for the configuration command line interface niceties )*

# Hardware requirements
* A decent computer to run machine learning training on
* An audio recording device, a cheap multi directional microphone will suffice
* Mouse and keyboard, for configuration purposes
* Eye tracker device ( if you want to use this program without a mouse )

# Installation

TODO - Fill this up so the installation isn't a pain to go through

Download the nircmd program and place it in the root folder of this project, in order to allow it to enable and disable the system volume.

In order to make use of the speech recognition part, you will have to enable the Windows Speech Recognition feature on your computer and train it to properly recognize your voice. 

In order to use an eye tracker, you will have to configure and train it to recognize your eyes. As well as install Project IRIS to easily toggle the mouse movement using eye tracking on and off, and it configure the smoothness of the movement.

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
Strategy - You can choose between

# Running the program

You can run the program by executing play.py . This process might crash if you haven't properly installed certain packages or configurated your models.

### Todos

 - Write the full installation part, if other people want to fool around with this prototype
 - Add examples of how to use the program
 - Seperate out the eyetracker and speech recognition parts, in order to allow the users not to have to install and configure all the different parts in order to get set up properly. 

License
----

MIT License

Copyright (c) 2019 Kevin te Raa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.