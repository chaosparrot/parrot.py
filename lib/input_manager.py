from config.config import REPEAT_DELAY, REPEAT_RATE
import time
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0
if (IS_WINDOWS == True):
    import pydirectinput
    pydirectinput.FAILSAFE=False
    pydirectinput.PAUSE = 0.0
import threading

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
    use_direct_keys = False
    
    # Used for input key up delays
    input_release_lag_ms = 0
    press_timings = {}
    input_release_thread = None
    
    def __init__(self, is_testing = False, repeat_delay=REPEAT_DELAY, repeat_rate=REPEAT_RATE, use_direct_keys=False, input_release_lag_ms=0):
        self.is_testing = is_testing
        self.repeat_delay = repeat_delay
        self.repeat_rate_ms = round(1000 / repeat_rate) / 1000
        
        # When we need to add an input delay to every key press ( because the game we are playing has a lot of input lag )
        # We start up a new thread to make sure the execution of the program goes as smooth as possible
        self.input_release_lag_ms = input_release_lag_ms
        if (self.input_release_lag_ms > 0):
            self.input_release_thread = threading.Thread(name='input_release_thread', target=input_release_thread, args=(self, self.input_release_lag_ms / 1000 ) )
            self.input_release_thread.setDaemon( True )
            self.input_release_thread.start()
        
        # Use DirectX keys - Needed in some programs that do not capture virtual keycodes
        self.use_direct_keys = use_direct_keys
        if (use_direct_keys == True):
            print("Using DirectX keycodes" )
        
        if( is_testing ):
            self.function_mappings['press'] = self.pressTest
            self.function_mappings['keyDown'] = self.keyDownTest
            self.function_mappings['keyUp'] = self.keyUpTest
            self.function_mappings['click'] = self.clickTest
            self.function_mappings['mouseDown'] = self.mouseDownTest
            self.function_mappings['mouseUp'] = self.mouseUpTest
        elif (self.use_direct_keys == True and IS_WINDOWS == True):
            self.function_mappings['press'] = self.pressActionDirect
            self.function_mappings['keyDown'] = self.keyDownActionDirect
            self.function_mappings['keyUp'] = self.keyUpActionDirect
            self.function_mappings['click'] = self.clickActionDirect
            self.function_mappings['mouseDown'] = self.mouseDownActionDirect
            self.function_mappings['mouseUp'] = self.mouseUpActionDirect
        else:
            self.function_mappings['press'] = self.pressAction
            self.function_mappings['keyDown'] = self.keyDownAction
            self.function_mappings['keyUp'] = self.keyUpAction
            self.function_mappings['click'] = self.clickAction
            self.function_mappings['mouseDown'] = self.mouseDownAction
            self.function_mappings['mouseUp'] = self.mouseUpAction

    def __del__(self):
        if(self.input_release_thread is not None):
            for key in self.press_timings:
                self.keyUp(key)

    def press( self, key ):
        if (self.input_release_lag_ms == 0):
            self.function_mappings['press'](key)
        elif (key not in self.press_timings ):
            self.press_timings[key] = time.time()
            self.keyDown(key)
        
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
            self.press( key )    
        
    def clickAction(self, button='left'):
        print( "----------> CLICKING " + button )        
        pyautogui.click( button=button )
        
    def mouseDownAction( self, button='left' ):
        print( "----------> HOLDING DOWN MOUSE " + button )
        pyautogui.mouseDown( button=button )
        
    def mouseUpAction( self, button='left' ):
        print( "----------> RELEASING MOUSE " + button )    
        pyautogui.mouseUp( button=button )
        
    # --------- ACTUAL PYDIRECTINPUT ACTIONS ---------
                
    def pressActionDirect(self, key):
        print( "----------> PRESSING " + key )
        pydirectinput.press( key )

    def keyDownActionDirect(self, key):
        print( "----------> HOLDING DOWN " + key )    
        pydirectinput.keyDown( key )
        
    def keyUpActionDirect(self, key):
        print( "----------> RELEASING " + key )
        pydirectinput.keyUp( key )
        
    def holdActionDirect( self, key ):    
        if( time.time() - self.last_key_timestamp > throttle ):
            self.last_key_timestamp = time.time()
            self.press( key )
        
    def clickActionDirect(self, button='left'):
        print( "----------> CLICKING " + button )        
        pydirectinput.click( button=button )
        
    def mouseDownActionDirect( self, button='left' ):
        print( "----------> HOLDING DOWN MOUSE " + button )
        pydirectinput.mouseDown( button=button )
        
    def mouseUpActionDirect( self, button='left' ):
        print( "----------> RELEASING MOUSE " + button )    
        pydirectinput.mouseUp( button=button )
        
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
        
def input_release_thread( inputManager, loop_delay):
    while(True):
        current_time = time.time()
        deleted_keys = []
        for key in inputManager.press_timings:
            if ( current_time - inputManager.press_timings[key] > loop_delay):
                inputManager.keyUp(key)
                deleted_keys.append(key)
        
        for key in deleted_keys:
            del inputManager.press_timings[key]

        time.sleep(loop_delay / 4)
    