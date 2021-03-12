# Connecting Talon Voice with Parrot
----

Talon Voice is a program that allows you to make your own voice commands, but as of the current version ( 0.1.5 ) it does not support more than two noises ( a pop and a hiss ).
As Parrot allows for any type of noise to be made, but lacks amazing speech recognition possiblities beyond Windows Speech Recognition, these two complement eachother fairly well.

Parrot aims to be Talon Voice compatible, as such, when major changes to the interprocess communications are made in Talon, Parrot will aim to be updated before the next major release of Talon is released.
If Talon Voice aims for personalized models in a later date, Parrot will aim to make the model that Talon uses for noises trainable and transferable between the two programs.
This will allow you ( the user ) to pick which program you like best, or if you want to continue using Talon and Parrot in conjunction with one another.

There is a channel inside of the Talon Voice slack ( http://talonvoice.slack.com/ ) called parrot that discusses the integration of Parrot and Talon. If you need help, it is best to ask there.

Set-up
-----

After you have both Talon and Parrot installed, trained your first model and have it tested to your needs, you can start integrating Talon and Parrot.
In the docs/examples/talon folder, you will find two files ( parrot_communication.py and parrot_communication.talon ) that you can use to interact with a running Parrot instance.
If you place these files inside of your Talon user folder, you should be able to use the voice commands laid out in parrot_communication.talon
That means you can control Parrot from inside Talon.

In order to send commands over to Talon, you can move the other file ( mode_pokey.py, named after Pokey Rule ( https://github.com/pokey ) that helped set up and pioneer the Parrot communication with Talon ) over to the lib/modes folder. 
In it, you will find an example pattern and some lines you can uncomment to send various commands over to Talon.

By default, the names of your patterns will get send over to Talon over its repl connection as a user action. Let's say you have the pattern name 'hiss', the following command will be sent to the repl: actions.user.hiss()

When you run Parrot and Talon together, the two will maintain a connection. If one of the programs is stopped and restarted, the other will pick up the connection next time it is restarted again.
In Talon, this is when the program fully boots up. in Parrot, this is whenever a sound is made and no open connection exists, it will try to connect to the repl once every 0.5 seconds if a sound is made.

Options and possibilities
-----

Combining Talon and Parrot comes with a bunch of advantages. In the base_talon_mode.py file, you will find a bunch of predefined methods that you can use to speed up your workflow.
Some examples include:
- Binding 'talon wake' and 'talon sleep' commands to a noise, I like using a bell
- Binding meta actions like 'repeat that', 'undo that' etc to a noise for quick repeating and undoing
- Binding mouse scrolling on a noise and have it scroll faster based on how loud a sound is
- Binding mouse clicks to a noise to make it go faster than uttering 'touch' or 'righty'

Inside a pattern, there is also a namespace reserved for Talon related actions.
For example, this is what happens if you add the following piece of code to a pattern:

```
    'talon': {
        'send': False,
		'throttle': 1.0
    }
```

This will not send the pattern name over to Talon as a user action, and when the pattern gets detected, Talon will be put into sleep mode for 1 second every time the noise is made.
When second is over, Talon will automatically be enabled again. This can be handy if you have a noise that Talon likes to respond to itself. 
I had a nasal sound 'N' that Talon would recognize as a command when I held it for a long time, with this throttle, Talon won't respond to it. 
If Parrot does crash during a sleep mode, you can always reawaken Talon using 'talon wake'

Trying out your own commands
-----

If you want to get more creative, you can send messages directly to the repl instead. You can use the write method available to the base_talon_mode to write a line directly towards Talon.
As you won't get any feedback like printing or output right now, it is best to test these commands first by opening the Talon repl first and testing the commands out before sending them from Parrot.

The repl program is located inside the bin folder within your Talon home directory.

Communication details
-----

As of the creation of this document, the best way to communicate with Talon is using the named pipes used in the repl program. This ensures a low latency connection, and gives you lots of freedom to try out different things.
In the future, this interprocess communication may change to a different format. But right now it should work for all 0.1.x versions of talon.