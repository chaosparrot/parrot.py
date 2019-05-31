# Recording sound files
----

In order to train a model, you need to record sounds first. You can do this by running `py settings.py` and pressing the [R] key, this will lead you through the steps neccesary for recording.

![Installing packages](media/settings-record.png)

This script will record sounds in seperate files of 50 milliseconds each and save them in your recordings folder ( data/recordings is the default place, which can be changed in the config/config.py file ). 
You have to be sure to record as little noise as possible. For example, if you are recording a bell sound, it is imperative that you only record that sound.
If you accidentally recorded a different sound, you can always delete the specific file from the recordings directory.

![Installing packages](media/settings-record-progress.png)

In order to make sure you only record the sound you want to record, you can alter the loudness setting at the start. I usuually choose a value between 1000 and 2000.
You can also trim out stuff below a specific frequency value. Neither the loudness or the frequency values I am using isn't actually an SI unit like dB or Hz, just some rough calculations which will go up when the loudness or frequency goes up.

During the recording, you can also pause the recording using SPACE or quit it using ESC. If you feel a sneeze coming up, or a car passes by, you can press these keys to make sure you don't have to prune away a lot of files.

I found that you need around a 1000 samples to get proper recognition of a sound. But you can try any amount and see if they recognize well.

[Step 2 - Training the model](TRAINING.md)