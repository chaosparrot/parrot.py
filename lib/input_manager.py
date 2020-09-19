from config.config import REPEAT_DELAY, REPEAT_RATE
import time
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

# Managers sending inputs to manipulate the keyboard or mouse, or to print out statements in testing mode
class InputManager:

    function_mappings = {
        'press': False,
        'keyDown': False,        
        'keyUp': False,
        'click': False,
        'mouseDown': False,
        'mouseUp': False,
    }
    
    special_keys = ['ctrl', 'shift', 'alt']
    
    toggle_keys = {
        'ctrl': False,
        'shift': False,
        'alt': False,
        'up': False,
        'down': False,
        'left': False,
        'right': False
    }

    key_hold_timings = {}
    is_testing = False
    
    def __init__(self, is_testing = False, repeat_delay=REPEAT_DELAY, repeat_rate=REPEAT_RATE):
        self.is_testing = is_testing
        self.repeat_delay = repeat_delay
        self.repeat_rate_ms = round(1000 / repeat_rate) / 1000
        
        if( is_testing ):
            self.function_mappings['press'] = self.pressTest
            self.function_mappings['keyDown'] = self.keyDownTest
            self.function_mappings['keyUp'] = self.keyUpTest
            self.function_mappings['click'] = self.clickTest
            self.function_mappings['mouseDown'] = self.mouseDownTest
            self.function_mappings['mouseUp'] = self.mouseUpTest
        else:
            self.function_mappings['press'] = self.pressAction
            self.function_mappings['keyDown'] = self.keyDownAction
            self.function_mappings['keyUp'] = self.keyUpAction
            self.function_mappings['click'] = self.clickAction
            self.function_mappings['mouseDown'] = self.mouseDownAction
            self.function_mappings['mouseUp'] = self.mouseUpAction
                
    def press( self, key ):
        self.function_mappings['press'](key)
        
    def keyDown( self, key ):
        self.function_mappings['keyDown'](key)
        
    def hold( self, key, repeat_rate_ms=0 ):
        if( repeat_rate_ms == 0 ):
            repeat_rate_ms = self.repeat_rate_ms

        if( key in self.toggle_keys.keys() ):
            if (self.toggle_keys[key] == False):
                self.keyDown( key )
                self.toggle_keys[ key ] = True
        else:
            if( key not in self.key_hold_timings ):
                self.key_hold_timings[key] = time.time()
                self.press(key)
            elif( time.time() - self.key_hold_timings[ key ] > self.repeat_delay ):
                self.key_hold_timings[ key ] += repeat_rate_ms
                self.press(key)
                
    def release_non_toggle_keys( self ):
        heldDownKeys = list(self.key_hold_timings)
        for key in heldDownKeys:
            if( key not in self.toggle_keys.keys() ):
                self.release( key )
            
    def release_special_keys( self ):
        for key in self.special_keys:
            if( self.toggle_keys[ key ] == True ):
                self.release( key )
        
    def release( self, key ):
        if( self.is_testing ):
            print( "-> RELEASING " + key )
        
        if( key in self.toggle_keys and self.toggle_keys[key] == True ):
            self.keyUp( key )
            self.toggle_keys[ key ] = False
        elif( key in self.key_hold_timings ):
            del self.key_hold_timings[key]

    def keyUp( self, key ):
        self.function_mappings['keyUp'](key)
        
    def click( self, button='left' ):
        self.function_mappings['click'](button)
        
    def mouseUp( self, button='left' ):
        self.function_mappings['mouseUp'](button)

    def mouseDown( self, button='left' ):
        self.function_mappings['mouseDown'](button)
                
    # --------- ACTUAL PYAUTOGUI ACTIONS ---------
                
    def pressAction(self, key):
        print( "----------> PRESSING " + key )
        pyautogui.press( key )

    def keyDownAction(self, key):
        print( "----------> HOLDING DOWN " + key )    
        pyautogui.keyDown( key )
        
    def keyUpAction(self, key):
        print( "----------> RELEASING " + key )    
        pyautogui.keyUp( key )
        
    def holdAction( self, key ):    
        if( time.time() - self.last_key_timestamp > throttle ):
            self.last_key_timestamp = time.time()
            self.cast_ability( key )    
        
    def clickAction(self, button='left'):
        print( "----------> CLICKING " + button )        
        pyautogui.click( button=button )
        
    def mouseDownAction( self, button='left' ):
        print( "----------> HOLDING DOWN MOUSE " + button )
        pyautogui.mouseDown( button=button )
        
    def mouseUpAction( self, button='left' ):
        print( "----------> RELEASING MOUSE " + button )    
        pyautogui.mouseUp( button=button )
        
    # --------- TEST METHODS FOR PRINTING ---------

    def pressTest(self, key):
        print( "-> Pressing " + key.upper() )

    def keyDownTest(self, key):
        print( "-> Holding down " + key.upper() )
        
    def keyUpTest(self, key):
        print( "-> Releasing " + key.upper() )
        
    def holdTest(self, key):
        print( "-> Pressing " + key.upper() )
        
    def releaseTest(self, key):
        print( "-> Releasing " + key.upper() )        
        
    def clickTest(self, button='left'):
        print( "-> Clicking " + button.upper() + " mouse button" )
        
    def mouseDownTest( self, button='left' ):
        print( "-> Holding down " + button.upper() + " mouse button" )
        
    def mouseUpTest( self, button='left' ):
        print( "-> Releasing " + button.upper() + " mouse button" )