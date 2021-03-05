# Connecting Parrot to other programs 
----

For advanced use cases, or for when you want to use the noise capabilities of Parrot with the speech recognition capabilities of other programs such as Talon Voice, it is possible to interact with Parrot.

The following interactions can be executed from outside of Parrot:
- Pausing and resuming the recording
- Stopping Parrot altogether
- Switching modes
- Switching classifiers

As of right now, Parrot needs to run in other to receive these commands

Talon Voice example
----- 

In the docs/examples/talon folder, you will find a pair of files that you can use to interact with a running Parrot instance.
If you place these files inside of your talon user folder, you should be able to use the voice commands laid out in parrot_communication.talon

In-depth explanation
-----

Parrot has an inter-process communication set up based around shared memory. This basically means that there is a shared memory block that can be manipulated to send data across.
All of the functionality is found in the file lib/ipc_manager.py .

You can also send data across to parrot to the free allocation block if you so desire.