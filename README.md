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
* Eye tracker device ( if )

# Installation

TODO - Fill this up so the installation isn't a pain to go through

Download the nircmd program and place it in the root folder of this project, in order to allow it to enable and disable the volume.

In order to make use of the speech recognition part, you will have to enable the Windows Speech Recognition feature on your computer and train it to properly recognize your voice. 

In order to use an eye tracker, you will have to configure and train it to recognize your eyes. As well as install Project IRIS to easily toggle the mouse movement using eye tracking on and off, and it configure the smoothness of the movement.

### Todos

 - Write the full installation part, if other people want to fool around with this prototype
 - Add examples of how to use the program

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