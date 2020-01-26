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

    def __init__(self, is_testing = False):
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
        
    def clickTest(self, button='left'):
        print( "-> Clicking " + button.upper() + " mouse button" )
        
    def mouseDownTest( self, button='left' ):
        print( "-> Holding down " + button.upper() + " mouse button" )
        
    def mouseUpTest( self, button='left' ):
        print( "-> Releasing " + button.upper() + " mouse button" )