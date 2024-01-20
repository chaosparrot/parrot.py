# Research

I have done a bunch of research, emperical and statistical, to see what works. While I am more often than not incorrect in my assumptions, there are a bunch of things we can rely on as assumptions / known variables during the development of Parrot.py and the problem space it envelops. Those assumptions / priori are shown below.

## Decibel priori

Because we are dealing with a 16 bit recording, the lowest dBFS level for absolute silence is -96 dBFS.
Thermal / environmental noise however adds a lot to this equation, and I personally haven't seem it ever drop the signal below -70 dBFS for longer periods of time. All our checks for -150 dBFS, while theoretical when all the values are zeroed, should thus be adjusted to at least -96.

When I adjusted a laptop microphone to have no noise amplification and a software volume of 1 on a scale of 100, I observed dBFS as low as -90, but never below that. Obviously, the higher the volume is set, and the higher the dB amplification is set, the higher the highest peak can go, but the lowest level never goes below -90.

The highest possible dBFS to be captured is 0, because all the bits are 1 in that case. When this is the case, we most likely have a case of clipping ( where the sound is scaled so much that it gets chopped off at its peak ). The data above this clipping threshold isn't recoverable.
The more values max out at 0 after one another, the higher the chance that clipping is occurring.

During a recording, a microphone can be turned off with a hardware button, this makes the dBFS value drop by a lot as environmental and thermal noise isn't recorded anymore. The microphone can also be hardware and software tweaked during a recording, though we do not want this to happen. This can be made clear to the user.

## Frequency priori

1. The recordings are done with a 16 kHz sample rate, this means the highest frequency we can capture is about 8 kHz due to the Nyquist frequency.
2. 16 kHz is the same sample rate used by telephone lines. This sample rate, while lower than CD quality, thus still keeps spoken words intelligible
3. Because a frame is 30 milliseconds long, the lowest frequency that can be captured is about 33 Hz.
4. Frames overlap by about 15 milliseconds each, this means frequencies from the previous frame can affect the ones in the current frame.
5. Shifts in frequency can be measured quickly and cheaply using zero-crossing rate, but we do not know the frequency composition this way.
6. In order to represent a sound, we use mel-frequency cepstrums. This transforms the sound into N values, where every value represents the loudness of one specific frequency band. These frequency bands are modeled after human hearing.
7. The windows used for the mel values are 15 ms long, overlapping by 5 milliseconds. For a frame with 30 ms, this means we have 4 MFCs. ( At 0 until 15, 5 until 20, 10 until 25 and 15 until 30 ms).
8. Because the window used is 15 ms, or half a frame, the lowest frequency that can be accurately captured in these buckets is 66 Hz.
9. In a single recording frame, the last MFCC and the first MFCC do not overlap.
10. In three recording frames ( named A, B and C ) we thus have 12 MFCs. A0, A5, A10 and A15 are the first frames, B0, B5, B10 and B15 are the frames after that, and C0, C5, C10 and C15 are the final frames.
11. A15 and B0 overlap completely, as well as B15 and C0. In theory this means we could cut off 5ms from the start of almost every recording and only calculate 3 MFCs per frame, and using the last MFC from the previous frame.
12. Because of the stride of 15 ms for every frame, we do not have any seams with windowing, because every part of the audio travels through the center of a 15ms stride. The only seams are at the very start and very end of the recording.
13. 20ms is generally the shortest amount of time used for frames, because the frequencies are most stable there ( <- source this, I read this somewhere but forgot where )

## Distribution priori

We do not know the distribution of the sound compared to the noise in a recording ahead of time. We also do not know the peak of a sound ahead of time. We do know that the loudest N percent of a recording is highly likely to encapsulate the desired sound, and the softest N percent of a recording is highly likely to encapsulate the environmental noise to keep out. There is also an unknown N percentage of the recording which contain environmental sounds ( keys being pressed to pause or resume for instance ) that is most likely less than the percentage of constant noise and the percentage of desired sound.

Previous experiments have shown that knowning the percentage of sound available, we can get pretty close to a single dBFS threshold that captures the sound while keeping most of the noise out within a single dB reliably.

## Behavioral priori

When people are recording parrot noises, we can assume a number of behaviors about the recording:

1. Within the first N seconds of the recording starting, the person has already made at least one valid sound.
2. The person recording tries to limit the amount of audible noise coming in, either by pausing the recording periodically, deleting segments with faulty sounds, or stopping the recording entirely.
3. The person tries to not record noises that aren't part of the one desired for that session.
4. The person tries to reach a decent amount of recorded data in as little time as possible.
5. The person inevitably has some breaks in the recordings where only environmental noise is heard.
These breaks occur either because the person needs to breathe, or they need to move their muscles in the starting position of making a sound again.
6. The person can vary their pitch, duration, loudness, and position from the microphone while still making the same sound.
7. The person is highly likely to record in a quiet office room or at home in front of a computer.
8. The recording is done using a built-in laptop microphone, a bluetooth microphone, a head set, or an on-desk microphone.
9. One or more microphones can be used during a single recording, but one single file always uses one microphone.

## Desired sound priori

1. The desired recording always rises above the noise floor in certain frequencies.
2. Stop consonants are generally the shortest sound, which have a duration of about 45 ms including the pause
( The duration of American-English stop consonants: an overview - Journal of Phonetics (1988) 16, 285-294 )
3. When the onset of a sound starts, it is only considered complete if the loudness of its frequencies have dropped back down below its original onset dB level.
4. A clear distinction in noise versus the desired sound can visibly be seen if a spectogram is shown from a recording session in Audacity.
5. Some desired sounds have more distinct frequency variety ( whistling in different notes ) than others ( clicking with your tongue ).
6. Some desired sounds have more duration variety ( sibilants, nasals ) than others ( pops, clicks, stop consonants ).
7. Almost all sounds can be varied in loudness and position from the microphone.
8. A sound earlier in the recording can be softer than a sound made later in the recording, but they should both be captured.
9. Given a known average dBFS or average mel frequency cepstrum, we can determine if a sound is closer to the desired sound, than to environmental noise, or an outlier.

## Environmental noise priori

1. Electrical noise is available throughout the entire recording.
2. Thermal noise is available throughout the entire recording.
3. Electrical and thermal noise can be modeled as Gaussian noise.
4. The amount of general noise is related to the quality and sensitivity of the microphone and whether or not it is close to metal and prone to vibrations from typing or other movements.
5. Environmental noise that does not happen throughout the entire recording ( breathing, wheezing, coughing ) cannot be modeled effectively, as it is unknown what frequencies are used. They can also be at the same loudness, or louder, than the desired sound.
6. Environmental noise most likely isn't similar to the desired sound frequency wise.

## Mel frequency priori

1. We use 40 filter banks in our MFCC. These correspond to smaller buckets at the start, and larger at the end, to approximately match the logarhymthic hearing of humans. The log from the filter bank is then taken, resulting in a list of log-Mels for every window.
2. The default output from our implementation standardizes over the log MELs, which scales and shifts the data to make the standard deviation 1 and centers the data around a mean of 0.
3. Standardization (Z-score normalization) has an advantage over normalization ( Scale normalization ) in that we do not need to know the min and max of the full signal ahead of time, so it is easier to apply piece-wise per frame without prior knowledge of the signal.
4. Standardization is also less effected by outliers than normalization. So it is helpful in training models.
5. Normalization is helpful when we want to maintain the scale in relation to the other features and other frames. When we are comparing one frame to another, normalization is prefered over standardization because the scale is kept the same across multiple frames.
6. Comparing two spectograms together is called "Spectral flux", doing this in the mel scale it is sometimes called the "Mel flux". To do this , the spectrums needs to be normalized on the same scale.

## Onset detection

1. For onset detection, instead of comparing two spectograms together, normally we only compare the mel bins that have increased positively. Doing this is called a "half-wave rectifier" function.
2. For onset detection, an 'onset detection function' is used to calculate a single value. This single value can either be thresholded ( If this value is above X, we detect an onset ) or checked using peak detection ( If this is the largest peak in N time, we consider it an onset ). The resulting list of values is called the "Onset strength envelope" where higher values denote a higher probability for an onset
3. The most likely ODF we can use is a combination of Mel flux comparisons, in combination with peak finding.
4. In order to make the peak finding and ODF more accurate, we can make use of adaptive whitening, which rescales the mel bins according to previously heard maxima within that bin using a PSP ( peak spectral profile ). An exponential decay is executed across this PSP, which makes the maximum values slowly decay in order to account for louder and softer parts of a recording. Because our recordings do not have any defined melody or softer parts, we should keep this decay fairly stringent in order to account for softer occurrences in between louder ones
( https://c4dm.eecs.qmul.ac.uk/papers/2007/StowellPlumbley07-icmc.pdf )
5. For peak finding, a solution using Spectral flux and a filter is proposed in "Evaluating the online capabilities of onset detection methods" ( https://zenodo.org/records/1416036 ). Another solution is proposed and is called "SuperFlux" ( http://phenicx.upf.edu/system/files/publications/Boeck_DAFx-13.pdf )
6. Slow onsets are harder to detect than instantanious onsets ( https://www.researchgate.net/profile/Eugene-Coyle/publication/4192852_Onset_detection_using_comb_filters/links/0912f51114bb0963af000000/Onset-detection-using-comb-filters.pdf ) so we should probably be content with just having a nicely fitted dBFS threshold to find these, and then mending the onset in order to find most of the sound that belongs here.
7. Because the shortest sound that is emperically made is about 50 ms, the maximum peak of an ODF within a frame of 50 ms before, and 50 ms after the onset frame, would be the best indicator. Because our frames are 30 ms with 15 ms windowed, we should probably use 30 ms before ( second frame in the past ) and 30 ms after ( second frame past the current in the future ) to have 90 ms of detection windows. The spectograms compared with one another should not overlap.
8. In an online setting, we cannot know the future, so just using the second frame in the past should work for peak finding.
9. A minimum threshold of δ needs to be found in order to make the online algorithm not pick out a peak every 100 ms. A value of 4 was shown to work decently for the "Approximant J" recording, but we can probably find a value after some experimentation with the data and other files that we have on hand.
10. For the second pass, when we know the full signal, we can probably find this value by finding the median over the peaks, in order to rule out false positives that seem to crop up a lot in online detection.
11. We can also apply backtracking to a local minima to refine the onsets in an online or offline environment ( "Creating music by listening" )