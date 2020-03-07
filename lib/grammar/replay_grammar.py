from dragonfly import Grammar, CompoundRule, Integer, Choice, Repetition, Optional
from pyautogui import press, typewrite
from config.config import *

quickCommands = {}
quickCommands["production statistics"] = "d"
quickCommands["income statistics"] = "i"
quickCommands["unit statistics"] = "u"
quickCommands["upgrade statistics"] = "g"
quickCommands["ay pee em statistics"] = "m"
quickCommands["toggle camera"] = "c"
quickCommands["view player"] = "1"
quickCommands["view opponent"] = "2"
quickCommands["view everyone"] = "e"
quickCommands["speed up"] = "+"
quickCommands["warp speed"] = "+++"
quickCommands["slow down"] = "-"
quickCommands["pause replay"] = "p"
quickCommands["resume replay"] = "p"

commandChoices = Choice( "quickcommand", quickCommands)

class ReplaySpeechCommand(CompoundRule):
    spec = "Replay <quickcommand>"
    extras = [commandChoices]
    callback = False
    
    def set_callback( self, callback ):
        self.callback = callback
    
    def _process_recognition(self, node, extras):
        press('esc')
        print( "REPLAY COMMAND - " + extras['quickcommand'] )
        typewrite( extras["quickcommand"], interval=0.1 )
        
        if( self.callback ):
            self.callback()

class ToggleEyetrackerCommand(CompoundRule):
    spec = "Replay (toggle|disable) eyetracker"
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
