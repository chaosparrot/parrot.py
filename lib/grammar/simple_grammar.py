from dragonfly import Grammar, CompoundRule, Integer, Choice, Repetition, Optional
from pyautogui import press, typewrite
from config.config import *

class SimpleSpeechCommand(CompoundRule, choices):
    spec = "<quickcommand>"
    extras = [new Choice( "quickcommand", choices)]
    callback = False
    
    def set_callback( self, callback ):
        self.callback = callback
    
    def _process_recognition(self, node, extras):
        typewrite( extras["quickcommand"], interval=0.1 )
        
        if( self.callback ):
            self.callback()