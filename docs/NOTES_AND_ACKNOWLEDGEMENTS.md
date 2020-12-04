# Research and notes
-----------

During the development of Parrot.py I came across some articles and blog bposts that might defend some of the choices made in terms of the audio bits.
Below I also have some emperical notes, which I do not have scientific backings for, but are none the less observations that might help someone else as well.

![Preference for 20-40 ms window duration in speech analysis by Kuldip K. Paliwal, James G. Lyons and Kamil K. Wojcicki](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.185.3622&rep=rep1&type=pdf)
This paper is one of the reasons why I chose a recording sound of 30 milliseconds. 

![SELU - Make FNNs Great Again by Elior Cohen](https://towardsdatascience.com/selu-make-fnns-great-again-snn-8d61526802a9)
This blog post, coupled with emperical evidence, provides the reason why I use SeLU over ReLU in my current architecture

Emperical notes 
------

- Normalization provides for a better accuracy. In previous versions of Parrot I had added the power value and the frequency to the data that I gave to the recognition models.
However, removing those values increased the validation accuracy by a full percentage point from 86.8 to 87.9 using a dataset containing 37 noises and with the models essentially the same.
Changing the pre-emphasis value in the MFCC, which boosts higher frequencies, from 0.97 to 0.5 improved accuracy from 86.9 to 87.2.
A later experiment using batch normalization on the input layer also improved the accuracy from 94.1 to 94.7. 
Currently I use the SeLU activation functions which also have normalizing properties.
In an experiment with a 10 layer DNN comparing ReLU to SeLU, ReLU topped out at 94.1 percent, while SeLU topped out at 95.2 percent.

- Dropout improved accuracy by a lot. Using a similar set up, the accuracy of the model went from 92.1 to 94.1 percent with a dropout of 0.1 . Clearly the regularization helped immensely.

- Making the temporal changes more visible in the audio data seems to be preferable to using more qualitative data. 
During the experiments I gradually changed the MFCC parameters to include more cepstrums, smaller window steps and lengths.
The increase in cepstrums from 13 to 20 increased accuracy from 87.9 to 91.6 percent.
Changing the window step from 10 milliseconds to 5 milliseconds increased accuracy from 86.8 to 87.2.
Later experiments with a window step of 3 milliseconds increased the accuracy of that current model from 91.6 to 92.1 percent
Changing the window length from 25 milliseconds to 20 milliseconds also improved accuracy, though I do not have data using comparable models.
All of this seems to point towards temporal changes being far more important than using larger window steps to more accurately encode the data.

- Making DNNs deeper improves accuracy at the cost of latency. I have found 6 DNNs to be a good middle ground so far to make sure not a lot of audio frames are skipped.

- Data augmentation doesn't seem to provide a large accuracy increase when compared to just adding more recorded audio. While it can be useful to give more data to a specific label, I would generally advise just recording more data as it proves to be better.
It might provide better accuracy when using different microphones or environments, but I haven't tested the models in different environments yet.

- I have done multiple experiments with convolutional layers, from fully convolutional to convolutional in front of some fully connected layers. 2D convolutional and 1D convolutional layers as well.
I have not been able to match the accuracy of the DNN models I have made so far, trailing behind by a couple of percentages. Perhaps I am doing something wrong or I should use a different input.
Some thoughts surrounding why this may be is related to the findings that the temporal nature of the audio is more important that the feature accuracy.
As convolutional layers are location-invariant, they might lose out on the temporal information, resulting in worse accuracies. I am no expert on the matter, so perhaps someone else can make CNNs work in the case of noises.

- Data quality is better than data quantity. Of course, with the right amount of patience while recording noises, you can achieve the best of both worlds.
It is best to set the threshold parameters well to only record a few samples for each uttering, because you wish to use the key presses on the onset of the sound, not on the echo.
Using a continuous noise dataset of 30 labels I was able to chop the labeled audio files into 30ms audio files required for learning in Parrot.py.
I used a random forest as an estimator of how well the models would work, and I segmented the audio files with different power thresholds to see their results.
With a threshold of 5k, the accuracy was only about 34%. Increasing the threshold to 10k improved that accuracy to 45%, and again increasing the threshold to 20k improved it to 82%
Using higher thresholds meant less background noise was added to the audio, and also that the labels would have less variance inside of them. While the number of samples decreased due to more being filtered out, the validation accuracy improved.

- Using less labels will improve accuracy as well, as the model does not need to learn a lot of differences in that manner. 
If there are labels that do not seem to recognize well, it might prove to be better to just leave them out.
However, in certain circumstances, having one label that isn't well recognized might paradoxically make it easier to recognize others, as it makes the model recognize other differences in the data.

- Using a random forest to train your noises can be an excellent indicator about how qualitative the data is. While neural nets provide better accuracies all around, they take a while longer to train.
A random forest is trained relatively quickly, and when accuracy is above 90ish percent, you can be pretty sure the noises were extracted from recordings properly.

- Ensembles are great for a number of reasons. They provide better accuracies than single neural nets, and they provide a measure of indecisiveness.
While neural nets with a soft max activation layer provide better accuracy in general, they tend to lean towards higher percentage outcomes even when they are incorrect.
Putting them in an ensemble of 3 allows each net to only contribute 33% of the outcome, and only 90%+ confidence can be achieved if there is consensus among the neural nets.
Thus, when unknown sounds are passed into the net, the odds of something being miscategorized over 90% is much lower than just using a single net, allowing for simpler thresholding.

- Random forests are useful for smaller amounts of noises, say below 20 different ones, however, when going past this number, they become gradually less useful.
It is advised to step into the domain of neural nets in cases where there are a lot of different noises to recognize.
During the development of the Starcraft mode, I was stuck at a use of about 26 noises while using a random forest.
Once I was able to move to neural nets, that number gradually increased to 37 for the current Hollow Knight mode.

- False negatives are better than false positives. While playing games, it is frustrating to have something not activate, but it can be corrected once the error is shown by simply repeating the noise and hoping the model recognizes the input this time.
However, when a model activates a incorrect input from a noise, it will most likely cause an unexpected action to occur that may or may not be reversable.
Examples of this include me activating my ultimate in Heroes of the Storm, abilities which have cooldowns upward of 100 seconds which cannot be reversed.
Or me missing a control group in starcraft, causing me to send a different army towards their inevitable doom.
Or me falling off of platforms into certain death because it miscategorized going left as going right in Hollow Knight.

- Tactile feedback is used more than you think. When holding down control or other keys using a keyboard, you implicitly know it is activated because you can feel the key pressed down.
However, when buttons are no longer in the picture, this feeling disappears and needs to be replaced. Adding a visual overlay near the vision of the user can provide cues that certain keys are held down, like the control key.

- It is better to drop a few audio frames if things take too long to process than to keep a buffer of them. Playing games tends to require a very snappy reaction time, and it's no use doing actions with a delay of 150 milliseconds because a buffer of audio frames has built up.
It is better to discard the buffer gradually in that case, which Parrot does by skipping every other audio frame if the buffer is larger than 2.

# Acknowledgements
-----------

Even though I am the main contributor, this project would not have gone this far without the help of other people.
This section aims to put a spotlight on them in no particular order. Most of these are from the wonderful community centered around Talon Voice, if Parrot.py is something you enjoy, this program will blow your mind.
Please consider donating some noises for that program with ![this link](https://noise.talonvoice.com). It greatly helps out the ecosystem of hands-free software.

The first user other than me ( I forgot their name, shame on me ) - For showing me that this program was usable for other people as well, as well as showing me that I should probably think twice about randomly changing things and dependencies as backwards compatibility is nice to have as well.

Various reddit users in the starcraft subreddit - For pointing out the confusing nature of the program output during demos, making it harder to follow which noises contribute to which actions. This resulted in me making the control visualisation that is visible in the hollow knight demo.

![feardragon64](https://github.com/feardragon64) - For briefly working on a Hands-Free starcraft project himself, and providing me with the courage to try my program out on the ladder versus actual players.

![lunixbochs](https://github.com/lunixbochs) - For pointing out computational optimizations with the audio pipeline and providing a slack for a wonderful community.

![jcaw](https://github.com/jcaw) - For helping out with brainstorming the deep learning bits, pointing towards normalization and SeLU activation functions and generally doing experiments with noises as well.
Also for making the recording script for Talon to help boost their dataset.

![okonomichiyaki](https://github.com/okonomichiyaki) - For testing out the documentation by starting from scratch and implementing their own interaction modes.

timotimo - For motivating me to start implementing audio conversion scripts and providing a continuous recording dataset from which I can do some research on.

You - For reading all the way down this page and showing interest in this program <3

# Showcases
------------

Playing Starcraft 2 at Diamond league level

https://www.youtube.com/watch?v=okwLAHQdSVI

Playing Heroes of the Storm against AI

https://www.youtube.com/watch?v=wSFDCgfGA9U

Playing Hollow Knight

https://www.youtube.com/watch?v=aG2qiFiOOYo