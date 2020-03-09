# Making your own interaction mode
----

The goal of this step is to make your own interaction model that does keyboard and mouse interactions when responding to your sounds and speech.
This step will require programming, but I've made this tutorial with the assumption of no prior knowledge. So it should at least be followable for a novice.
You will need to have a trained model before we can interact with it however, so make sure you have followed the previous steps. Or simply use the dummy classifier ( which only classifies noise and silence )

In the examples directory, there are files labeled mode_tutorial_a.py - These are complete files with comments and explanations about what they do.
You're free to copy and paste them in the lib/modes/ directory, and if you're running into issues, you can always copy back the original in the examples directory over the altered one in the lib/modes directory.

Running an interaction mode for testing ( uses mode_tutorial_a.py )
----

First, copy over the mode_tutorial_a.py file from the examples directory over to the lib/modes directory. This is an empty interaction mode that won't react to anything.
If you need a clean configuration, you can also copy over the config file in the examples directory to the config directory.

We can run our program using the `play.py` command.
This will automatically run the default classifier and the default mode in the config/config.py command. 
You can always change this to match your own trained model and classifier.
Remember, you can always pause or quit the program by pressing SPACE or ESC respectively.
If you encounter a crash during this command, make sure you have followed the installation process and make sure you haven't skipped any steps.

For now, we will run the `py play.py` command with a few bits added to them. 
Adding ` -t` will run the program in testing mode. In this mode - No key presses or mouse clicks will be performed, but they will be printed in your terminal.
Adding ` -m mode_tutorial_a` will run the program with the mode_tutorial_a file placed in the lib/modes directory. Currently, because mode_tutorial_a is empty, it won't do much but listen for sounds.
Adding ` -c dummy` will run the program using the dummy classifier which only recognizes sound and noise. Of course, you can replace dummy with your own trained model to use that model.
If any of these additional bits are omitted, the program will revert to the defaults set in the config/config file.

As you can see, nothing is actually being done now when you are running the program. We're going to change that in the next step.

Keyboard and mouse interaction ( uses mode_tutorial_b.py )
----

Inside the mode_tutorial_b.py file, there are a number of lines between four # signs that denote the different kind of basic interactions that you are able to do.
For the keyboard, you can press and hold down keys. And for the mouse you can drag, left and right click. 

Copy the mode_tutorial_b file over and remove one of the # signs in front of a line to test it out. You can run the testing command to safely see if it is registering a command.
`Press` instantly presses a key whenever a sound is detected, but this will cause a lot of repeat keystrokes rapidly after one another.
`Hold` holds a key down and sends press events, trying to simulate your keyboard. That means whenever hold is run, it will press one key first, and only after half a second, will repeat the key after half a second. 
Just like your regular keyboard! This is generally a safer option to take. 
`Leftclick` presses your left mouse button, while `rightclick` presses the right one.
`Drag_mouse` holds down your left mouse button, enabling dragging over things like text or units.

You will notice that once you turn silent, things like mouse dragging and holding down keys will automatically stop. This is something that is done for you in the lib/modes/base_mode file.
The only keys that are kept pressed down are CTRL, SHIFT or ALT if they were run with hold. You can release those special keys using the `release` function.

More interactions, such as scrolling and holding down the middle mouse button, are possible using the pyautogui library that you installed when setting up Parrot.py. 
[View the pyautogui documentation and possible keyboard keys here](https://pyautogui.readthedocs.io/en/latest/keyboard.html)
You can also look in the lib/input_manager file for examples on how pyautogui is used.

Detection strategies ( uses mode_tutorial_c.py )
----

Now we are going to connect our sounds to our keyboard inputs.