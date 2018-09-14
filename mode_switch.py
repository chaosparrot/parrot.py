from dragonfly import Grammar, CompoundRule
import pythoncom
from time import sleep
import pyautogui

# Voice command rule combining spoken form and recognition processing.
class TwitchModeRule(CompoundRule):
    spec = "Twitchmode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
		# SWITCH TO TWITCH MODE
        pyautogui.typerwite("Twitchmode")

class BrowseModeRule(CompoundRule):
    spec = "Browsemode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
		# SWITCH TO BROWSE MODE
        pyautogui.press("r") 

class GameModeRule(CompoundRule):
    spec = "GameMode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
		# SWITCH TO GAME MODE
        pyautogui.press("r") 

class DraftModeRule(CompoundRule):
    spec = "GameMode"                  # Spoken form of command.
    def _process_recognition(self, node, extras):   # Callback when command is spoken.
		# SWITCH TO GAME MODE
        pyautogui.press("r") 
		
		 
# Create a grammar which contains and loads the command rule.
grammar = Grammar("example grammar")                # Create a grammar to contain the command    rule.
grammar.add_rule(TwitchModeRule())                     # Add the command rule to the grammar.
grammar.add_rule(BrowseModeRule())                     # Add the command rule to the grammar.
grammar.add_rule(GameModeRule())                     # Add the command rule to the grammar.
grammar.add_rule(DraftModeRule())                     # Add the command rule to the grammar.

grammar.load()                                      # Load the grammar.