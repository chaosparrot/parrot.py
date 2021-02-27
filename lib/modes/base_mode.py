from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from lib.input_manager import InputManager
from time import sleep
from subprocess import call
from lib.system_toggles import toggle_eyetracker, toggle_speechrec
from lib.pattern_detector import PatternDetector
from config.config import *
import os
# import pythoncom
from lib.overlay_manipulation import update_overlay_image

# Quadrants
TOPLEFT = 1
TOPMIDDLE = 2
TOPRIGHT = 3
CENTERLEFT = 4
CENTERMIDDLE = 5
CENTERRIGHT = 6
BOTTOMLEFT = 7
BOTTOMMIDDLE = 8
BOTTOMRIGHT = 9

class BaseMode:
    quadrant3x3 = 0
    quadrant4x3 = 0
    speech_commands = {}
    patterns = {}
    toggles = {}
    use_direct_keys = False
    input_release_lag = 0
                
    def __init__(self, modeSwitcher, is_testing=False, repeat_delay=REPEAT_DELAY, repeat_rate=REPEAT_RATE):
        self.inputManager = InputManager(is_testing=is_testing, use_direct_keys=self.use_direct_keys, input_release_lag_ms=self.input_release_lag * 1000)
        self.mode = "regular"
        self.modeSwitcher = modeSwitcher
        self.detector = PatternDetector(self.patterns)
        self.pressed_keys = {}
        self.should_drag = False
        self.ctrlKey = False
        self.shiftKey = False
        self.altKey = False
        
        if( SPEECHREC_ENABLED == True ):
            from dragonfly import Grammar
            from lib.grammar.simple_grammar import SimpleSpeechCommand
            import pythoncom
            self.grammar = Grammar("Simple")
            self.simpleCommandRule = SimpleSpeechCommand(self.speech_commands, callback=self.toggle_speech)
            self.grammar.add_rule( self.simpleCommandRule )
        
    def start( self ):
        update_overlay_image( "default" )
        toggle_eyetracker()
        
    def exit( self ):
        if( self.mode == "speech" ):
            self.toggle_speech()
    
        update_overlay_image( "default" )
        toggle_eyetracker()
                    
    def handle_input( self, dataDicts ):
        self.detector.tick( dataDicts )
        self.quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
        self.quadrant4x3 = self.detector.detect_mouse_quadrant( 4, 3 )
        
        if( self.detect_silence() ):
            self.stop_drag_mouse()
            self.inputManager.release_non_toggle_keys()
                
        # Recognize speech commands in speech mode
        if( self.mode == "speech" ):
            pythoncom.PumpWaitingMessages()
            self.handle_speech( dataDicts )
            
        # Regular quick command mode
        elif( self.mode == "regular" ):
            self.handle_sounds( dataDicts )
            
        return self.detector.tickActions
                
    def handle_speech( self, dataDicts ):
        print( "No speech handler" )
        return
        
    def handle_sounds( self, dataDicts ):
        print( "No sound handler" )
        return
        
    # Toggle between variables
    # If the value is a list, turn them on in sequence after another
    def toggle( self, value ):
        if (isinstance(value, list)):
            turned_on_index = -1
            for index, key in enumerate(value):
                if (key not in self.toggles):
                    self.toggles[key] = False
                elif (self.toggles[key] == True):
                    turned_on_index = index
                    self.toggles[key] = False
                    
            next_index = turned_on_index + 1
            if (next_index >= len(value)):
                next_index = 0

            self.toggles[value[next_index]] = True
        else:
            if (value not in self.toggles ):
                self.toggles[value] = False
                
            self.toggles[value] = not self.toggles[value]
    
    def enable( self, value ):
        if (isinstance(value, list)):
            for index, key in enumerate(value):
                self.toggles[key] = True
        else:
            self.toggles[value] = True

    def disable( self, value ):
        if (isinstance(value, list)):
            for index, key in enumerate(value):
                self.toggles[key] = False
        else:
            self.toggles[value] = False
        
    def detect( self, key ):
        if (key in self.toggles):
            return self.toggles[key]
    
        return self.detector.detect( key )
        
    def detect_silence( self ):
        return self.detector.detect_silence()        
        
    def drag_mouse( self ):
        self.toggle_drag_mouse( True )

    def stop_drag_mouse( self ):
        self.toggle_drag_mouse( False )
                
    def leftclick( self ):
        self.inputManager.click(button='left')

    def rightclick( self ):
        self.inputManager.click(button='right')
        
    def press( self, key ):
        self.inputManager.press( key )
        
    def hold( self, key, repeat_rate_ms=0 ):
        self.inputManager.hold( key, repeat_rate_ms )
        
    def release( self, key ):
        self.inputManager.release( key )
        
    def release_special_keys( self ):
        self.inputManager.release_special_keys()
        
    def toggle_speech( self ):
        if( self.mode != "speech" ):
            self.mode = "speech"
            self.grammar.load()
            print( "--- TOGGLING SPEECH RECOGNITION ON" )
        else:
            self.mode = "regular"
            self.grammar.unload()
            print( "--- TOGGLING SPEECH RECOGNITION OFF" )
        toggle_speechrec()

    # Drag mouse for selection purposes
    def toggle_drag_mouse( self, should_drag ):
        if( self.should_drag != should_drag ):
            if( should_drag == True ):
                self.inputManager.mouseDown()
            else:
                self.inputManager.mouseUp()
                
        self.should_drag = should_drag
        
    # Detect when the cursor is inside an area
    def detect_inside_area( self, x, y, width, height ):
        return self.detector.detect_inside_minimap( x, y, width, height )

    def update_overlay( self ):
        if( not( self.ctrlKey or self.shiftKey or self.altKey ) ):
            update_overlay_image( "default" )
        else:
            modes = []
            if( self.ctrlKey ):
                modes.append( "ctrl" )
            if( self.shiftKey ):
                modes.append( "shift" )
            if( self.altKey ):
                modes.append( "alt" )
                
            update_overlay_image( "mode-%s" % ( "-".join( modes ) ) )        