# Using Talon Voice with Parrot
----

Talon Voice is a program that allows you to make your own voice commands, but as of the current version ( 0.1.5 ) it does not support more than two noises ( a pop and a hiss ).
As Parrot allows for any type of noise to be made, but lacks amazing speech recognition possiblities beyond Windows Speech Recognition, these two complement eachother fairly well.

I have collaborated with TalonVoice to make it possible to have first party support. For this you need to have access to the beta of TalonVoice. Instructions on how to set it up can be found there. But the gist of it is the following:

- First, train a AudioNet model, the resulting pkl file is supported by Talon Voice, and name it model.pkl
- Second, make a directory called 'parrot' inside your Talon home folder
- Third, place the model.pkl file inside the folder
- Fourth, Create a patterns.json file inside the parrot folder, this will contain the patterns you would otherwise have inside of parrot modes in a JSON format. Do note that the format is not a one to one match as of right now.
  It is suggested you read up on how to change the patterns to suit your needs from the PDF document inside the Talon Voice slack
- Fifth, retrieve the parrot_integration.py file from the #beta channel on the talonvoice slack and put it in your user folder
- And finally, add the noises to your .talon files like this

```
parrot(noise_name): mouse_click(0)
```

When tweaking your noise patterns within Talon Voice, it is highly recommended to use the visualisation tool [Parrot Tester](https://github.com/rokubop/parrot_tester) made by [Rokubop](https://github.com/rokubop). It allows you to more granularly look at how noises are detected, and what thresholds you should use to make them more effective.

There is a channel inside of the Talon Voice slack ( http://talonvoice.slack.com/ ) called #parrot that discusses the integration of Parrot and Talon. 
If you need help, it is best to ask there. I will try to answer your questions personally if I happen to be there.