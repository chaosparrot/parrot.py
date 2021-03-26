# Linux installation guide
-----------

Step one - Installing required programs
----------

Open up a terminal and see if python is installed by running the following snippet ```python --version```. If it shows 3.8, you're good to go!
Otherwise you can install python using your local package manager, make sure you install 3.8, as some libraries don't work well with 3.9 yet.

The two dependencies that I needed to install on Linux for my testdrive were 'TK' and 'portaudio'. 

Step two - Download and extract the zipfile from this github repository
---------------

![Extracting parrot.py](media/install-parrotpy.png)

Now download the zip of this github repository and save it somewhere on your computer. It doesn't need to be in program files or anything, it can just be on your Desktop.
Extract the files in another folder and you can start testing the settings menu.
Now, make sure to go to your Parrot.py location, in my case that is the desktop. So it should be in `cd ~/Desktop/parrot.py-master` 
Make sure that you can see the requirements-posix.txt file if you run the `ls` command, that means you're in the right place!

Step three - Setting up your pipenv
---------

While it is possible to installthe python packages directly using pip install, it is recommended to make a pipenv first because it will be easier to remove stuff.
To do this, run the following commands:

```
pip install pipenv
pipenv --python 3.8
```

Step four - Installing all the packages
---------

With your terminal in the right place, it is time to download and install all the required libraries. Run the following code in your terminal to install them all.

```bash
pipenv install -r requirements-posix.txt
```

After this, enter the pipenv using ```pipenv shell```
Now you can run the following command to see if everything works

```bash
python settings.py
```

If any errors occur, try opening the requirements-posix.txt file and installing each line seperately and run again.
Exiting the pipenv is as simple as entering ```exit``` in the shell or pressing Ctrl-D

Step five - Running parrot.py
--------

In order to run Parrot.py from your pipenv, you first have to enter it using ```pipenv shell```, after this, you can run ```python play.py```.

Additional optional steps can be found on the [Windows installation page](INSTALLATION.md).