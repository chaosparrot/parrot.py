from dragonfly import Grammar, CompoundRule
from detection_strategies import single_tap_detection, loud_detection, medium_detection
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position
from time import sleep
from subprocess import call
from system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
import os
import pythoncom
from mode_twitch import *
from mode_browse import *
from mode_youtube import *
from mode_switch import *

class SwitchMode:

	def __init__(self):
		self.mode = "regular"
		# Create a grammar which contains and loads the command rule.
		grammar = Grammar("example grammar")                # Create a grammar to contain the command    rule.
		grammar.add_rule(TwitchModeRule())                     # Add the command rule to the grammar.
		grammar.add_rule(YoutubeModeRule())                     # Add the command rule to the grammar.
		grammar.add_rule(BrowseModeRule())                     # Add the command rule to the grammar.
		grammar.add_rule(GameModeRule())                     # Add the command rule to the grammar.
		grammar.add_rule(DraftModeRule())                     # Add the command rule to the grammar.
		grammar.load()                                      # Load the grammar.		

	def start( self ):
		mute_sound()
		toggle_speechrec()
		
	def handle_input( self, dataDicts ):
		pythoncom.PumpWaitingMessages()
		sleep(.1)
		
	def exit( self ):
		turn_on_sound()
		toggle_speechrec()

# Voice command rule combining spoken form and recognition processing.
class TwitchModeRule(CompoundRule):
    spec = "Twitchmode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
        toggle_speechrec()
        x = ModeSwitcher()
        x.switchMode(TwitchMode())
		
# Voice command rule combining spoken form and recognition processing.
class YoutubeModeRule(CompoundRule):
    spec = "Youtubemode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
        # SWITCH TO YOUTUBE MODE
        toggle_speechrec()		
        x = ModeSwitcher()
        x.switchMode(YoutubeMode())

class BrowseModeRule(CompoundRule):
    spec = "Browsemode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
        # SWITCH TO BROWSE MODE
        toggle_speechrec()
        x = ModeSwitcher()
        x.switchMode(BrowseMode())

class GameModeRule(CompoundRule):
    spec = "GameMode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
        # SWITCH TO GAME MODE
        toggle_speechrec()
        x = ModeSwitcher()
        x.switchMode(BrowseMode())

class DraftModeRule(CompoundRule):
    spec = "DraftMode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
        # SWITCH TO GAME MODE
        toggle_speechrec()
        x = ModeSwitcher()		
        x.switchMode(BrowseMode())
