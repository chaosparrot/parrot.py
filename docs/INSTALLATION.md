# Installation guide
-----------

For this program to work, you must atleast have access to a microphone and a Windows machine that can run Python 3.6 32 bit.

Do note that this guide only details how to set up the base installation of this program ( i.e. being able to run the program without random crashes ). 
There are no ready-made audio recognition models added to this installation, you will have to make those yourself using the settings menu.
You will also have to change the modes to your own needs.

Step one - Installing python
----------

![Installing python](media/install-python.png)

This can be done by following the [Python installation link](https://www.python.org/downloads/release/python-360/) and selecting your desired way of installing - I used the Windows x86 executable installer option.
Currently, one of the packages ( pyaudio ) relies on Python 3.6, so that is the easiest version to install from.
Make sure you have the checkbox 'Add Python 3.6 to PATH' enabled and pick the Install Now option. 

If you install from a higher version that will be supported with fixes longer than 2021, for now you will have to install some .whl files manually in the next step.

Step two - Installing all the packages
---------

![Installing packages](media/install-libs.png)

Now that you have python installed, you can use it to download packages. Open a command line program ( Search for cmd in your Windows search box ) and test if your python is installed properly.
This can be done by typing 'python -v', this will display the version of your current python installation.

If all else is well, install the packages below by running the following lines of code in your command line program: 

```bash
pip3 install requirements.txt
```

Step three - Download and extract the zipfile from this github repository
---------------

![Extracting parrot.py](media/install-parrotpy.png)

Now download the zip of this github repository and save it somewhere on your computer. It doesn't need to be in program files or anything, it can just be on your desktop.
Extract the files in another folder and you can start testing the settings menu.
Navigate to the directory in your command line tool by using `cd YOUR_PATH_TO_PARROT_PY_DIRECTORY` and then run `py settings.py`.
If it shows the options menu, you should be able to run all the recording, training and analysis tools.

Step four - Training Windows Speech recognition
---------------

Before you can make use of the speech recognition part, you will have to train Windows Speech recognition on your microphone and your voice.
You can do this by opening up windows speech recognition and following their installation instructions. 
You can also use Dragon Speech Naturally if you want more proper dictation possibilities, but for short commands, windows speech recognition functions just fine.

With all these steps followed, you can run the `py play.py` command on your command line tool to run the program. But first, you will have to tweak your settings to match your needs. 

Optional - Download NIRCMD.exe and place it inside the root folder
----------------

This executable allows the program to more easily toggle the system volume. If you don't need this functionality, it isn't required.

Optional - Install the eyetracker software
---------------

To be able to move your mouse cursor without using your hands, you must have access to an eyetracker and a way to map the cursor roughly to the place where you are looking.
For this, I use the Tobii Eye Tracker 4C and a seperate program called Project IRIS. The latter can be trialed for about a month before you have to purchase it.
I haven't tried any other eye tracker, but this one seems just fine for this program.

Follow their installation instructions and make sure you can toggle your mouse cursor following by pressing a key - I have F4 configured to toggle it

Optional - Install PyTorch for advanced neural network usage
----------------

I personally saw a big improvement when I switched to neural networks running in an ensemble. 
Pytorch offers a vast array of machinelearning techniques that might be useful to get even more recognition strength out of your generated models. 

But setting up Pytorch might be complicated because you will also need CUDA support and a decent graphics card if you intend to make large models with it.
If you're just playing around for the first time - You can just stick to the regular installation until you require more accuracy.

When the time comes you need more accuracy, you can download the pytorch version here: https://pytorch.org/get-started/locally/ 
Make sure you select Python and pip for installation. After installing, make sure you also install audiomentations with the command below


```bash
pip3 install audiomentations;
```

Optional - Install FFMPEG for recording file conversion
----------------

It is handy to be able to convert already recorded audio files in case you want to tweak the channels or rate after having recorded the audio files.
For this reason, FFMPEG is required if you desire to convert existing files.
If you already have it installed, simply point to the place where it is installed inside the config/config.py file.

Installing ffmpeg on windows is as easy as downloading it and unzipping it somewhere on your computer. The executable should be inside the bin folder in there.
By default, the program assumes the ffmpeg executable is placed in ffmpeg/bin inside parrot.
Therefore, it is recommended to move the whole ffmpeg folder over to the parrot root directory and rename the ffmpeg root directory name to ffmpeg. 
That way it should work out of the box without having to go into configuration.