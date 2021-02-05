# Upgrade guide
-----------

The best way to upgrade Parrot.PY to make sure nothing breaks is to download this repository as a zipfile and extract it as a seperate folder to your current version.
Then to copy over the data directory and change the config/config.py to your previous values.

Upgrading from before 0.10.0
----

Note that because in the versions before 0.10.0 there was little concern for backwards compatibility, your current models might break when run in the 0.10.0 and above version.
Your recorded data might also not match up with the length of the files when recorded now. 
That means you might have to rerecord your audio and retrain your models.

My sincerest apologies for this inconvenience. As of version 0.10.0, there is a much better system in place that should avoid these problems in the future.

Version 0.10.0
-----
* Added BaseMode and new and improved pattern detector
* Fixed length of audio recordings and added a source file for complete recording
* Added improved feature engineering strategies
* Added audio conversion menu for advanced usecases
* Removed some former models that are no longer in use
* Implemented a structure for the classifiers to make upgrading less prone to breaking on updating the Parrot.PY version
* Improved documentation and tutorials