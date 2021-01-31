# Patterns

Patterns are basically just fancy ways of saying when does the program recognize it should detect something. Or in other words, if we want to press down a button, what sound should trigger that, and how does that sound like?
For the purpose of explanation, let's say we have a simple model that can recognize four sounds: Hissing, tongue clicking, whistling and silence.

### Thresholds

The first threshold you will deal with is the output of your trained model: The percentages of what it thinks the sound is. In our example model, the output of a single frame of sound would be something like this:
Hissing: **89%**
Tongue clicking: **2%**
Whistling: **8%**
Silence: **0%**

What you can do to trigger a button based on hissing, would be a pattern that looks something like this:

```
{
	'name': 'action_one',
	'sounds': ['Hissing'],
	'threshold': {
		'percentage': 85
	}
}
```

> 'When the model detects a hissing sound with 85% confidence or higher, we detect action_one'.

This is just fine to start off with, however, you will quickly find some problems. As the model isn't that sophisticated, it will recognize some sounds that it doesn't know as hissing.
Maybe you were just sliding a mouse over a mouse mat, or you were brushing your hand through your hair, and the model might think that is a hissing sound, because it doesn't know those other sounds.

The easiest fix for that is the 'power' and 'intensity' thresholds, which are basically just indications of how loud the sound needs to be. I would recommend using the 'power' threshold, as it is a more accurate representation of loudness.
When we change the threshold bit in the above example to this:
```
	'threshold': {
		'percentage': 85,
		'power': 10000
	}
```
It will only detect action_one when the sound is louder than 10.000 . Loudness is affected by how loud the microphone is set up, and how close the microphone is to the source of the sound.
In the case of brushing over your hair now, the sound will be very quiet, and thus not detect 'action_one'. This allows you to filter out a lot of different sounds that the microphone might pick up.

Another threshold is 'frequency', which is a measurement of the tone of the sound frame. Let's make a new action for the whistling sound, action_two.

```
{
	'name': 'action_two',
	'sounds': ['Whistling'],
	'threshold': {
		'percentage': 80,
		'frequency': 55
	}
}
```

> 'When the whistling sound is detected with 80% confidence, and the calculated frequency is above 55, detect action_two'. 

This allows you to filter out sounds that the model might think of as whistling, but are of a lower tone.

When you are dealing with a sound that gets misrecognized a lot, there is one more threshold to try out, the 'times' threshold.
If we add that to action_two, this might look something like this:
```
'threshold': {
	'percentage': 80,
	'frequency': 55,
	'times': 4
}
```

> 'Only when I have detected four sound frames with a confidence of 80% and a frequency of 55 and higher, will I detect action_two'. 

The audio model we have created might sometimes just detect the whistling sound randomly because of some sound it hasn't heard before.
This usually only happens for a frame or two, but that might be enough to press a button in a game that has severe consequences when pressed at the wrong time, like opening your inventory in the middle of a sword fight. 
The times threshold is the best stopgap for this, but this comes at a cost. Instead of the action being recognized immediately after about 20 milliseconds, it will only get triggered in 65 milliseconds ( 15 milliseconds per sound frame, with 5 milliseconds of latency ).
I have used this a lot with the finger snapping sound, as it gets misrecognized a lot.

### Lower thresholds

Not only can you detect when something is above a certain threshold, you can also check if something is below one. This comes in handy if we want to assign two different actions to the same sound.
Let's take the whistling sound as an example, what if we wanted an action for a low whistle, and a higher tone whistle, without having to retrain our model on two different sounds again? We'll just add a 'below_frequency' threshold.

```
{
	'name': 'action_three',
	'sounds': ['Whistling'],
	'threshold': {
		'percentage': 80,
		'below_frequency': 55
	}
}
```

> 'When the confidence of the whistle sound is over 80% and the frequency detected is below 55, detect action_three'. 

Because action_two is detected at a frequency of 55 or higher, we can now press different buttons with a different tune of whistle!
The lower thresholds you can use are 'below_power', 'below_intensity', 'below_percentage' and 'below_frequency'.

### Continual sounds

Some sounds are better used as long held sounds, like hissing, while other sounds are best used as short bursts like tongue clicking. You cannot extend a tongue click for a full second, but you can easily do that with a hiss!
However, as you continue a hiss, the sound slowly changes. Maybe you're running out of air, or maybe you're using slightly less pressure on the sound. Maybe your tongue gets in the way. There can be all kinds of reasons, but when the sound changes, the model will become less confident in it.
Let's extend action_one, the hissing sound, with a continual threshold to mitigate this.

```
{
	'name': 'action_one',
	'sounds': ['Hissing'],
	'threshold': {
		'percentage': 85,
		'power': 10000
	},
	'continual_threshold': {
		'percentage': 30,
		'power': 3000
	}
}
```

This is getting a bit more complicated, but here's the gist of this pattern: 

> 'When the hissing sound is above 85% confidence and above 10.000 power, detect action_one. 
> However, only stop detecting action_one when the sound frames after this fall below 30% confidence and 3000 power.'

This allows you to take into account the changing sound that might follow the initial hiss, making it less likely for the action to suddenly stop. You can even omit the percentage threshold in the continual_threshold bit and just go with the power threshold alone.
This will keep detecting action_one until the sound gets softer, allowing you to go for way longer. Doing this comes at a cost that any sound after that, even if it is a completely different sound, will get detected as action_one. But you can still tweak it around and see what works.

### Throttling

The detections happen roughly 60 times a second. If you have a button press linked to a pattern that gets detected for a full second, you will get 60 button presses.
Even for short sounds it might press the button multiple times for a single utterance. Let's say we have bound tongue clicking to the left mouse button, instead of the tongue click pressing the mouse button once, it might go for double clicks or even triple clicks.
For this, you can make use of throttles. 

```
{
	'name': 'action_four',
	'sounds': ['Tongue clicking'],
	'threshold': {
		'percentage': 95,
		'power': 30000
	},
	'throttle': {
		'action_four': 0.3
	}
}
```

> 'When tongue clicking is over 95% confidence and the power is above 30.000, detect action_four. 
> After this, wait 0.3 seconds, or 300 milliseconds before we can detect action_four again'

Now we have successfully made sure our tongue clicking sound only clicks the mouse button once per utterance. But when you do rapid tongue clicking, you might utter the same sound twice in that 300 milliseconds timeframe.
In the example above, you would only have pressed the left mouse button once. So the best throttle value is roughly the time it takes for a single utterance, which you might have to tune a bit.

But not only can you throttle your own action, you can throttle other actions as well. Let's say we change the throttle around to this:

```
'throttle': {
	'action_four': 0.3,
	'action_two': 1.0
}
```

> 'After the detection of this action, wait 300 milliseconds before we can detect action_four again, and wait one second before we can detect action_two again'

This comes in handy if you notice some misrecognitions happening right after you make a sound like tongue clicking. Maybe it recognizes a high whistle sound somewhere in the echo for some reason and it detects action_two without you actually whistling.
Adding a specific throttle for another action like action_two would take away that misdetection.

Throttling should be avoided when you are using continual thresholds, as a throttle skips the follow up detections altogether until that time is over.

### Combining sounds and detecing untrained sounds

Not only can you hang an action on single sound labels, but you can also combine them. Let's say we want to add another sound that the model doesn't know of, shushing. When we shush in our example model, it might output something like this

Hissing: **54%**
Tongue clicking: **1%**
Whistling: **45%**
Silence: **0%**

To make this into a pattern, we can write something like this

```
{
	'name': 'action_five',
	'sounds': ['Hissing', 'Whistling'],
	'threshold': {
		'percentage': 90,
		'power': 20000
	}
}
```

> 'When hissing and whistling combined are above 90% confidence, and the power is above 20.000, detect action_five

Now we have succesfully added a new sound despite not having trained our model to it. This technique can also be used if you have a pair of sounds that often gets recognized together, like tongue clicking and finger snapping, into a single pattern that gets detected better.
But, this does give an extra challenge. Now, even if we hiss for over 90% confidence, both action_one and action_five will be detected at the same time.

In order to make this new combined sound more specific, we can add the 'ratio' and 'below ratio' threshold.

Let's say we know that the shushing sound makes the percentages of whistling between 30 and 60 percent, and the same goes for hissing. We can add something like this

```
'threshold': {
	'percentage': 90,
	'ratio' 0.5,
	'below_ratio': 2
}
```

> 'When hissing and whistling combined are above 90% confidence, and dividing the percentage of Hissing with the percentage of Whistling gives us a number between 0.5 and 2, detect action_five'

To understand how it works, its probably the easiest to see some examples of sound frames and what actions they detect in different outputs

| Sound frame | Hissing percent | Whistling percent | Combined percentage | ratio | detected action_one | detected action_five |
|-|-|-|-|-|-|-|
| #1 | 35 | 60 | 95 | 0.58 | No | Yes |
| #2 | 55 | 30 | 89 | 1.83 | No | No |
| #3 | 90 | 5 | 95 | 18 | Yes | No |
| #4 | 54 | 38 | 92 | 1.42 | No | Yes |

As you can see, we have successfully made sure action_one and action_five never overlap detection wise

If you want to get fancy, you can even put these 'ratio' and 'below_ratio' thresholds in the continual thresholds area, but so far I haven't found a use for that yet

### Conclusion

That about sums up all the different kinds of patterns you can make and what they are useful for. This might expand in the future, but so far I have been able to play multiple games with this relatively basic amount of detection operations.
Combining this with mouse position detections can bring the amount of different actions you can do up tremendously, and I could easily map all the different hotkeys you need in an RTS like starcraft with a model that just recognized about 24 sounds
You just have to be creative and experiment in how you set up the patterns.