from dragonfly import Grammar, CompoundRule
import pythoncom
from time import sleep
import pyautogui
from mode_twitch import *
from mode_browse import *
from mode_youtube import *
from system_toggles import toggle_speechrec

# Voice command rule combining spoken form and recognition processing.
class TwitchModeRule(CompoundRule):
    spec = "Twitchmode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
        x = SwitchMode()
        x.switchMode(TwitchMode())
		
# Voice command rule combining spoken form and recognition processing.
class YoutubeModeRule(CompoundRule):
    spec = "Youtubemode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
        # SWITCH TO YOUTUBE MODE
        x = SwitchMode()
        x.switchMode(YoutubeMode())

class BrowseModeRule(CompoundRule):
    spec = "Browsemode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
        # SWITCH TO BROWSE MODE
        x = SwitchMode()
        x.switchMode(BrowseMode())

class GameModeRule(CompoundRule):
    spec = "GameMode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
		# SWITCH TO GAME MODE
        pyautogui.press("r") 

class DraftModeRule(CompoundRule):
    spec = "DraftMode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
		# SWITCH TO GAME MODE
        pyautogui.press("r") 
		
		 
# Create a grammar which contains and loads the command rule.
grammar = Grammar("example grammar")                # Create a grammar to contain the command    rule.
grammar.add_rule(TwitchModeRule())                     # Add the command rule to the grammar.
grammar.add_rule(YoutubeModeRule())                     # Add the command rule to the grammar.
grammar.add_rule(BrowseModeRule())                     # Add the command rule to the grammar.
grammar.add_rule(GameModeRule())                     # Add the command rule to the grammar.
grammar.add_rule(DraftModeRule())                     # Add the command rule to the grammar.
grammar.load()                                      # Load the grammar.

class SwitchMode(object):
	__instance = None
	__currentMode = None
	
	def __new__(cls):
		if SwitchMode.__instance is None:
			SwitchMode.__instance = object.__new__(cls)
		return SwitchMode.__instance

	def listenMode( self ):
		while( True ):
			pythoncom.PumpWaitingMessages()
			sleep(.1)
		
	def switchMode( self, nextMode ):
		toggle_speechrec()
		print( nextMode )
		if( SwitchMode.__currentMode is None ):
			SwitchMode.__currentMode = nextMode
			nextMode.start()
		else:
			SwitchMode.__currentMode.exit()
			SwitchMode.__currentMode = nextMode
			SwitchMode.__currentMode.start()
			

x = SwitchMode()
x.listenMode()