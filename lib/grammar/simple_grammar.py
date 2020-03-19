from dragonfly import Grammar, CompoundRule, Integer, Choice, Repetition, Optional
from pyautogui import press, typewrite
from config.config import *

class SimpleSpeechCommand(CompoundRule):
    callback = False
    
    def __init__( self, choices, callback=False ):
        print( choices )
        self.extras = [Choice( "quickcommand", choices)]
        self.spec = "<quickcommand>"
        self.callback = callback
        CompoundRule.__init__(self)
    
    def _process_recognition(self, node, extras):
        for command in extras['quickcommand']:
            if( command == 'exit' and self.callback ):
                self.callback()
            else:
                print( " -> PRESSING " + command )            
                if( INPUT_TESTING_MODE == False ):
                    press( command )