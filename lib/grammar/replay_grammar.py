from dragonfly import Grammar, CompoundRule, Integer, Choice, Repetition, Optional
from pyautogui import press, typewrite
from config.config import *

quickCommands = {}
quickCommands["show production"] = "d"
quickCommands["show income"] = "i"
quickCommands["show units"] = "u"
quickCommands["show actions"] = "m"
quickCommands["lock on"] = "c"
quickCommands["follow me"] = "1"
quickCommands["follow my vision"] = "1"
quickCommands["follow me and lock on"] = "1c"
quickCommands["follow my vision and lock on"] = "1c"
quickCommands["follow my opponents vision"] = "2"
quickCommands["follow my opponents vision and lock on"] = "2c"
quickCommands["follow them"] = "2"
quickCommands["follow them and lock on"] = "2c"
quickCommands["follow everyone"] = "e"
quickCommands["speed up"] = "="
quickCommands["engage warp speed"] = "==="
quickCommands["slow down"] = "-"
quickCommands["pause the game"] = "p"
quickCommands["resume the game"] = "p"

commandChoices = Choice( "quickcommand", quickCommands)

class ReplaySpeechCommand(CompoundRule):
    spec = "Abathur <quickcommand>"
    extras = [commandChoices]
    callback = False
    
    def set_callback( self, callback ):
        self.callback = callback
    
    def _process_recognition(self, node, extras):
        press('esc')
        typewrite( extras["quickcommand"], interval=0.1 )
        
        if( self.callback ):
            self.callback()

class ToggleEyetrackerCommand(CompoundRule):
    spec = "Abathur toggle eyetracker"
    extras = [commandChoices]
    callback = False
    
    def set_callback( self, callback ):
        self.callback = callback
    
    def _process_recognition(self, node, extras):
        press(EYETRACKING_TOGGLE)
        
        if( self.callback ):
            self.callback()

class QuitReplayCommand(CompoundRule):
    spec = "Abathur quit replay"
    extras = [commandChoices]
    callback = False
    
    def set_callback( self, callback ):
        self.callback = callback
    
    def _process_recognition(self, node, extras):
        press('esc')
        press('f10')
        press('q')
        
        if( self.callback ):
            self.callback()
