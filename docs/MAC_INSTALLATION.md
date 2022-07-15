# Mac installation guide
-----------

Step one - Installing required programs
----------

Open up a Terminal window and see if you have Homebrew installed by running the following command
```
brew --version
```

If it isn't returning a version, you need to install it. Follow the instructions here: [Homebrew installation](https://docs.brew.sh/Installation)

After you have installed homebrew, you need to install Python 3.8 and portaudio. This can be done using the following commands

```
brew install python@3.8
brew install portaudio
```

Test if your python version is correct by typing ```python --version``` in your Terminal, if it shows 3.8, you're good to go!

On the M1, some issues can occur while installing portaudio. There are a number of possible fixes that are outlined in these links:
- [Unable to install pyaudio on M1 Mac](https://stackoverflow.com/questions/68251169/unable-to-install-pyaudio-on-m1-mac-portaudio-already-installed)
- [MacOS Brew install libsndfile but still not found](https://stackoverflow.com/questions/70737503/macos-brew-install-libsndfile-but-still-not-found)

Step two - Download and extract the zipfile from this github repository
---------------

![Extracting parrot.py](media/install-parrotpy.png)

Now download the zip of this github repository and save it somewhere on your computer. It doesn't need to be in program files or anything, it can just be on your Desktop.
Extract the files in another folder and you can start testing the settings menu.
Now, make sure to go to your Parrot.py location, in my case that is the desktop. So it should be in `cd ~/Desktop/parrot.py-master` 
Make sure that you can see the requirements-posix.txt file if you run the `ls` command, that means you're in the right place!

Step three - Installing all the packages
---------

Now that you have your Terminal in the right place, it is time to download and install all the required libraries. Run the following code in your Terminal to install them all.

```bash
pip install -r requirements-posix.txt
```

Now you can run the following command to see if everything works

```bash
python settings.py
```

If any errors occur, try opening the requirements-posix.txt file and installing each line seperately and run again.

Step four - Enabling permissions
---------------

Parrot requires access to your Microphone, and the ability to press keys on your behalf. That means Mac requires some additional granted permissions to the Terminal.
The easiest way to grant these permissions is by just going through the recording instructions. You will most likely get a pop up which asks for permission to use your Microphone.

When you have your model trained and your mode set, running play.py and making it press a key will automatically open up a permission pop up for Accessibility access as well. Both should be granted for Parrot to run.

Additional optional steps can be found on the [Windows installation page](INSTALLATION.md).