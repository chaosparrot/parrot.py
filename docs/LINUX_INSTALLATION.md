# Mac installation guide
-----------

Step one - Installing required programs
----------

Open up a terminal and see if python is installed by running the following snippet ```python --version```. If it shows 3.8, you're good to go!
Otherwise you can install python using your local package manager, make sure you install 3.8, as some libraries don't work well with 3.9 yet.

The two dependencies that I needed to install on Linux for my testdrive were 'TK' and 'portaudio'. 
You also need a llvmlite version 10 to be installed, as version 11 does not seem to be properly supported for the dependencies. 

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

Additional optional steps can be found on the [Windows installation page](INSTALLATION.md).