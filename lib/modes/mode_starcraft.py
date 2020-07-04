from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from lib.input_manager import InputManager
from time import sleep
from subprocess import call
from lib.system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
from lib.pattern_detector import PatternDetector
from config.config import *
import os
import pythoncom
from lib.overlay_manipulation import update_overlay_image
from lib.grammar.chat_grammar import *
from lib.grammar.replay_grammar import *

class StarcraftMode:
            
    def __init__(self, modeSwitcher, is_testing=False):
        self.inputManager = InputManager(is_testing=is_testing)
        if( SPEECHREC_ENABLED == True ):
            self.grammar = Grammar("Starcraft")
            self.chatCommandRule = ChatCommandRule()
            self.chatCommandRule.set_callback( self.toggle_speech )
            self.replayCommandRule = ReplaySpeechCommand()
            self.toggleEyetracker = ToggleEyetrackerCommand()
            self.quitReplayCommand = QuitReplayCommand()
            self.quitReplayCommand.set_callback( self.toggle_speech )
            self.grammar.add_rule( self.chatCommandRule )
            self.grammar.add_rule( self.replayCommandRule )            
            self.grammar.add_rule( self.toggleEyetracker )
            self.grammar.add_rule( self.quitReplayCommand )
            
    
        self.mode = "regular"
        self.modeSwitcher = modeSwitcher
        self.patterns = {
            'select': {
                'strategy': 'continuous',
                'sound': 'sibilant_s',
                'percentage': 95,
                'intensity': 1400,
                'lowest_percentage': 50,
                'lowest_intensity': 1000,
                'throttle': 0
            },
            'rapidclick': {
                'strategy': 'continuous_power',
                'sound': 'general_thrill_thr',
                'percentage': 80,
                'lowest_percentage': 40,                
                'power': 20000,
                'lowest_power': 15000,                
                'throttle': 0
            },
            'click': {
                'strategy': 'frequency_threshold',
                'sound': 'click_alveolar',
                'percentage': 90,
                'above_frequency': 58,
                'power': 20000,
                'throttle': 0.2
            },
            'movement': {
                'strategy': 'frequency_threshold',
                'sound': 'sound_whistle',
                'percentage': 80,
                'below_frequency': 54,
                'power': 23000,
                'throttle': 0.3
            },
            'click_after_movement': {
                'strategy': 'frequency_threshold',
                'sound': 'sound_whistle',
                'percentage': 80,
                'above_frequency': 54,
                'power': 23000,
                'throttle': 0.3
            },
            'secondary_movement': {
                'strategy': 'rapid_power',
                'sound': 'sound_finger_snap',
                'percentage': 90,
                'power': 100000,
                'throttle': 0.3
            },
            'control': {
                'strategy': 'rapid_power',
                'sound': 'vowel_oh',
                'percentage': 80,
                'below_frequency': 40,
                'ratio': 0.01,
                'power': 20000,
                'throttle': 0.2
            },
            'secondary_control': {
                'strategy': 'combined_power',
                'sound': 'sibilant_z',
                'secondary_sound': 'fricative_v',                
                'percentage': 90,
                'power': 20000,
                'ratio': 0,
                'throttle': 0.2
            },
            'shift': {
                'strategy': 'rapid_power',
                'sound': 'sibilant_sh',
                'percentage': 90,
                'power': 18000,
                'throttle': 0.4
            },
            'alt': {
                'strategy': 'rapid_power',
                'sound': 'fricative_v',
                'percentage': 95,
                'power': 200000,
                'throttle': 0.4
            },
            'camera': {
                'strategy': 'combined_power',
                'sound': 'vowel_y',
                'secondary_sound': 'vowel_u',
                'ratio': 4,
                'percentage': 85,
                'power': 15000,
                'throttle': 0.25
            },
            'camera_secondary': {
                'strategy': 'combined_power',
                'sound': 'vowel_eu',
                'secondary_sound': 'vowel_y',
                'percentage': 60,
                'ratio': 0,
                'power': 20000,
                'throttle': 0.18
            },            
            'first_ability': {
                'strategy': 'combined',
                'sound': 'vowel_ow',
                'secondary_sound': 'vowel_u',
                'ratio': 0.3,
                'percentage': 90,
                'intensity': 1000,
                'throttle': 0.3
            },            
            'second_ability': {
                'strategy': 'rapid_power',
                'sound': 'vowel_ae',
                'percentage': 90,
                'power': 25000,
                'throttle': 0
            },
            'third_ability': {
                'strategy': 'rapid_power',
                'sound': 'approximant_r',
                'percentage': 90,
                'power': 45000,
                'throttle': 0
            },
            'r': {
                'strategy': 'rapid_power',
                'sound': 'fricative_f',
                'percentage': 90,
                'power': 20000,
                'throttle': 0.4
            },
            'grid_ability': {
                'strategy': 'combined_continuous',
                'sound': 'general_vowel_aa',
                'secondary_sound': 'vowel_ah',
                'ratio': 0,
                'percentage': 90,
                'intensity': 1500,
                'lowest_percentage': 12,
                'lowest_intensity': 900
            },
            'numbers': {
                'strategy': 'combined_power',
                'sound': 'vowel_iy',
                'secondary_sound': 'approximant_j',
                'ratio': 0,
                'percentage': 80,
                'power': 25000,
                'throttle': 0.16
            },
            'numbers_secondary': {
                'strategy': 'combined_power',
                'sound': 'vowel_y',
                'secondary_sound': 'vowel_ih',
                'percentage': 60,
                'ratio': 0,
                'power': 20000,
                'throttle': 0.16
            },
            'menu': {
                'strategy': 'rapid_power',
                'sound': 'sound_call_bell',
                'percentage': 80,
                'power': 100000,
                'throttle': 0.5
            }
        }
        
        self.detector = PatternDetector(self.patterns)

        self.KEY_DELAY_THROTTLE = 0.4
        
        self.pressed_keys = []
        self.should_follow = False
        self.should_drag = False
        self.last_control_group = -1
        self.ability_selected = False
        self.last_ability_selected = None
        self.ctrlKey = False
        self.shiftKey = False
        self.altKey = False
        self.hold_down_start_timer = 0
        self.hold_down_key_timer = 0
        self.last_key_timestamp = 0        
        
        self.hold_key = ""

    def toggle_speech( self, with_enter=True ):
        self.release_hold_keys()
        
        if( self.mode != "speech" ):
            self.mode = "speech"
            self.grammar.load()
            if( with_enter ):
                self.inputManager.press('enter')
        else:
            self.mode = "regular"
            self.grammar.unload()
        toggle_speechrec()
        
    # Used in case the speech recognition is triggered accidentally
    def reset_mode( self ):
        if( self.mode == "speech" ):
            self.grammar.unload()
            toggle_speechrec()
        self.inputManager.press("esc")
        self.detector.add_tick_action( "Esc" )
        self.mode = "regular"
        
    def start( self ):
        mute_sound()
        toggle_eyetracker()
        update_overlay_image( "default" )
        
    def cast_ability( self, ability ):
        self.press_ability( ability )
        self.ability_selected = True
    
    def hold_shift( self, shift ):
        if( self.shiftKey != shift ):
            if( shift == True ):
                self.inputManager.keyDown('shift')
                self.shiftKey = shift            
                self.update_overlay()                
            else:
                self.inputManager.keyUp('shift')
                self.shiftKey = shift
                self.update_overlay()
    
    def hold_alt( self, alt ):
        if( self.altKey != alt ):
            if( alt == True ):
                self.altKey = alt            
                self.update_overlay()
                self.detector.deactivate_for( 'first_ability', 0.1 )
                self.detector.deactivate_for( 'second_ability', 0.1 )
            else:
                self.altKey = alt
                self.update_overlay()
                self.detector.deactivate_for( 'first_ability', 0.3 )
                self.detector.deactivate_for( 'second_ability', 0.15 )
    
    def hold_control( self, ctrlKey ):
        if( self.ctrlKey != ctrlKey ):
            if( ctrlKey == True ):
                self.inputManager.keyDown('ctrl')
                self.ctrlKey = ctrlKey
                self.update_overlay()
            else:
                self.inputManager.keyUp('ctrl')
                self.ctrlKey = ctrlKey
                self.update_overlay()
        
    def release_hold_keys( self ):
        self.ability_selected = False
        self.hold_control( False )
        self.hold_shift( False )
        self.hold_alt( False )
        self.update_overlay()
    
    def handle_input( self, dataDicts ):
        self.detector.tick( dataDicts )
                
        # Always allow switching between speech and regular mode
        if( self.detector.detect( "menu" ) ):
            self.release_hold_keys()
            
            quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
            if( quadrant3x3 == 9 ):
                self.press_ability( 'f10' )
            elif( quadrant3x3 == 2 ):
                self.release_hold_keys()
                self.toggle_speech( False )
            elif( quadrant3x3 == 1 ):
                self.reset_mode()
            elif( quadrant3x3 == 7 ):
                self.mode = "ignore_commands"
            elif( quadrant3x3 == 3 ):
                self.release_hold_keys()
                self.toggle_speech()
            else:
                self.press_ability( 'esc' )
                self.detector.add_tick_action( "Esc" )
                
            self.update_command_file(dataDicts)
                    
            return self.detector.tickActions            
        # Recognize speech commands in speech mode
        elif( self.mode == "speech" ):
            pythoncom.PumpWaitingMessages()
            
            return self.detector.tickActions            
            
        # Regular quick command mode
        elif( self.mode == "regular" ):
            self.handle_quick_commands( dataDicts )

        self.update_command_file(dataDicts)
            
        return self.detector.tickActions
    
    def handle_quick_commands( self, dataDicts ):
        # Early escape for performance
        if( self.detector.detect_silence() ):
            self.drag_mouse( False )
            self.hold_down_start_timer = 0
            return
            
        if( self.detector.detect_below_threshold( 800 ) ):
            self.hold_down_start_timer = 0
            
        # Selecting units
        rapidclick = self.detector.detect("rapidclick")
        selecting = self.detector.detect( "select" )
        if( self.ability_selected and selecting ):
            self.inputManager.click(button='left')
            self.ability_selected = False
            
            # Clear the throttles for abilities
            self.detector.clear_throttle('camera')
            self.detector.clear_throttle('first_ability')
            self.detector.clear_throttle('second_ability')
            self.detector.clear_throttle('third_ability')
            self.detector.deactivate_for('select', 0.3)
        elif( rapidclick ):
            if( self.last_ability_selected == 'first' ):
                self.cast_ability_throttled('z', 0.03)
            elif( self.last_ability_selected == 'second' ):
                self.cast_ability_throttled('x', 0.03)
            elif( self.last_ability_selected == 'third' ):
                self.cast_ability_throttled('c', 0.03)
            
            # Prevent some misclassifying errors when using the thr sound
            self.detector.deactivate_for( 'control', 0.3 )
            self.detector.deactivate_for( 'click', 0.1 )
            self.detector.deactivate_for( 'grid_ability', 0.3 )
            self.ability_selected = False
        else:
            self.drag_mouse( selecting )
            
        ## Click after attacking
        if( self.detector.is_throttled('movement') and self.detector.detect("click_after_movement") ):
            self.inputManager.click(button='left')
            self.detector.add_tick_action( "Lclick" )

            self.ability_selected = False
        
        ## Press Grid ability
        if( self.detector.detect("grid_ability") and not rapidclick ):
            quadrant4x3 = self.detector.detect_mouse_quadrant( 4, 3 )
            if( time.time() - self.hold_down_start_timer > self.KEY_DELAY_THROTTLE ):
                self.use_ability_throttled( quadrant4x3, 0.025 )
                self.release_hold_keys()
            
            if( self.hold_down_start_timer == 0 ):
                self.hold_down_start_timer = time.time()
                
            self.detector.deactivate_for( 'control', 0.15 )
            self.detector.deactivate_for( 'secondary_control', 0.15 )
            self.detector.deactivate_for( 'movement', 0.15 )
        
        if( selecting ):
            self.ability_selected = False
            self.hold_control( False )

        elif( self.detector.detect( "click" ) ):
        
            # Cast selected ability or Ctrl+click
            if( self.detect_command_area() or self.ability_selected == True or self.ctrlKey == True or self.altKey == True or ( self.shiftKey and self.detect_selection_tray() ) ):
                self.inputManager.click(button='left')
                self.detector.add_tick_action( "left click" )
            else:            
                self.inputManager.click(button='right')
                self.detector.add_tick_action( "right click" )

            self.detector.deactivate_for( 'grid_ability', 0.4 )
            self.detector.deactivate_for( 'secondary_movement', 0.4 )
            self.detector.deactivate_for( 'movement', 0.3 )            
                
            # Release the held keys - except when shift clicking units in the selection tray ( for easy removing from the unit group )
            if( not( self.shiftKey and self.detect_selection_tray() ) ):
                self.release_hold_keys()
            
        # CTRL KEY holding
        elif( self.detector.detect( "control" ) ):
            self.hold_control( True )
            self.detector.add_tick_action( "CTRL" )
        elif( self.detector.detect( "secondary_control" ) ):
            self.hold_control( True )
            self.detector.deactivate_for( 'select', 0.2 )
            self.detector.deactivate_for( 'camera', 0.2 )
            self.detector.add_tick_action( "CTRL" )            
            
        # SHIFT KEY holding / toggling
        elif( self.detector.detect( "shift" ) ):
            self.hold_shift( not self.shiftKey )
            if (self.shiftKey):
                self.detector.add_tick_action( "SHIFT" )
    
        # ALT KEY holding / toggling
        elif( self.detector.detect( "alt" ) ):
            self.hold_alt( not self.altKey )
            if (self.shiftKey):
                self.detector.add_tick_action( "ALT" )


        ## Primary movement options
        elif( self.detector.detect( "movement" ) ):
            self.cast_ability( 'k' )
            self.detector.deactivate_for( 'control', 0.4 )
        
            self.hold_shift( False )
        ## Secondary movement options
        elif( self.detector.detect( "secondary_movement" ) ):
            quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
            if( quadrant3x3 == 1 ):
                self.cast_ability( 'p' )
                self.hold_shift( False )
            elif( quadrant3x3 == 3 ):
                self.press_ability( 'h' )
                self.hold_shift( False )

        ## Press Q
        elif( self.detector.detect( "first_ability" ) ):
            self.ability_selected = True
            self.detector.clear_throttle('rapidclick')
            self.last_ability_selected = 'first'
            
            self.inputManager.press( 'q' )
            self.detector.add_tick_action( "Q" ) 
            self.detector.deactivate_for( 'control', 0.25 )

            
        ## Press W
        elif( self.detector.detect( "second_ability") ):
            self.ability_selected = True            
            self.detector.clear_throttle('rapidclick')
            self.last_ability_selected = 'second'
            
            if( time.time() - self.hold_down_key_timer > self.KEY_DELAY_THROTTLE ):
                self.press_ability_throttled( 'w', 0.1 )
            
            if( self.hold_down_key_timer == 0 ):
                self.hold_down_key_timer = time.time()
        ## Press E
        elif( self.detector.detect( "third_ability") ):
            print( "THIRD ABILITY!" )
            self.ability_selected = True            
            self.detector.clear_throttle('rapidclick')
            self.last_ability_selected = 'third'
            
            if( time.time() - self.hold_down_key_timer > self.KEY_DELAY_THROTTLE ):
                self.press_ability_throttled( 'e', 0.1 )
            
            if( self.hold_down_key_timer == 0 ):
                self.hold_down_key_timer = time.time()
        ## Press R ( Burrow )
        elif( self.detector.detect( "r") ):
            self.last_ability_selected = 'third'
        
            self.inputManager.press( 'r' )
            self.detector.add_tick_action( "R" )
            
        ## Move the camera
        elif( self.detector.detect( "camera" ) ):
            quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
            self.camera_movement( quadrant3x3 )
            self.hold_control( False )
            self.hold_shift( False )
            self.hold_alt( False )
        elif( self.ctrlKey == True and self.detector.is_throttled('control') and self.detector.detect("camera_secondary") ):
            quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
            self.camera_movement( quadrant3x3 )
            self.hold_control( False )
            self.hold_shift( False )
            self.hold_alt( False )
            
            self.detector.deactivate_for('camera', 0.3)
            self.detector.deactivate_for('numbers', 0.3)
        ## Press control group ( only allow CTRL and SHIFT )
        elif( self.detector.detect( "numbers" ) ):        
            quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
            self.use_control_group( quadrant3x3 )
            
            self.hold_alt( False )                
            self.hold_control( False )
            self.hold_shift( False )
            self.detector.deactivate_for('camera', 0.3)
        elif( ( ( self.ctrlKey == True and self.detector.is_throttled('secondary_control') ) or self.shiftKey == True ) and self.detector.detect("numbers_secondary") ):
            quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
            self.use_control_group( quadrant3x3 )
            
            self.hold_alt( False )                
            self.hold_control( False )
            self.hold_shift( False )
            self.detector.deactivate_for('camera', 0.3)
            self.detector.deactivate_for('numbers', 0.3)
        else:
            self.hold_down_key_timer = 0
            
        return        
    
    def use_control_group( self, quadrant ):
        if( quadrant == 1 ):
            self.press_ability('1')
        elif( quadrant == 2 ):
            self.press_ability('2')
        elif( quadrant == 3 ):
            self.press_ability('3')
        elif( quadrant == 4 ):
            self.press_ability('4')
        elif( quadrant == 5 ):
            self.press_ability('5')
        elif( quadrant == 6 ):
            self.press_ability('6')            
        elif( quadrant == 7 ):
            self.press_ability('7')
        elif( quadrant == 8 ):
            self.press_ability('8')            
        elif( quadrant == 9 ):
            self.press_ability('9')
            
        self.last_control_group = quadrant
        
    def use_ability( self, quadrant ):
        if( quadrant == 1 ):
            self.press_ability('q')
        elif( quadrant == 2 ):
            self.press_ability('w')
        elif( quadrant == 3 ):
            self.press_ability('e')
        elif( quadrant == 4 ):
            self.press_ability('r')
        elif( quadrant == 5 ):
            self.press_ability('a')
        elif( quadrant == 6 ):
            self.press_ability('s')            
        elif( quadrant == 7 ):
            self.press_ability('d')
        elif( quadrant == 8 ):
            self.press_ability('f')            
        elif( quadrant == 9 ):
            self.press_ability('z')
        elif( quadrant == 10 ):
            self.press_ability('x')
        elif( quadrant == 11 ):
            self.press_ability('c')
        elif( quadrant == 12 ):
            self.press_ability('v')
            
    def use_ability_throttled( self, quadrant, throttle ):
        if( time.time() - self.last_key_timestamp > throttle ):
            self.last_key_timestamp = time.time()
            self.use_ability( quadrant )
        
    def press_ability( self, key ):
        self.inputManager.press( key )
        self.detector.add_tick_action( key.upper() )
        self.release_hold_keys()
        
    def cast_ability_throttled( self, key, throttle ):
        if( time.time() - self.last_key_timestamp > throttle ):
            self.last_key_timestamp = time.time()
            self.inputManager.press( key )
            self.detector.add_tick_action( key )
        
    def press_ability_throttled( self, key, throttle ):
        if( time.time() - self.last_key_timestamp > throttle ):
            self.last_key_timestamp = time.time()
            self.press_ability( key )
        
    def camera_movement( self, quadrant ):
        ## Move camera to kerrigan when looking above the UI
        if( quadrant == 1 ):
            self.inputManager.press( "f1" )
            self.detector.add_tick_action( "F1" )
        elif( quadrant == 2 ):
            self.inputManager.press( "f2" )
            self.detector.add_tick_action( "F2" )            
        elif( quadrant == 3 ):
            self.inputManager.press( "f3" )
            self.detector.add_tick_action( "F3" )
        elif( quadrant == 4 ):
            self.inputManager.press( "f5" )   
            self.detector.add_tick_action( "F5" )
        elif( quadrant == 5 ):
            self.inputManager.press( "backspace" )
            self.detector.add_tick_action( "backspace" )
        ## Camera hotkeys
        elif( quadrant == 6 ):
            self.inputManager.press( "f6" )
            self.detector.add_tick_action( "F6" )            
        
        ## Camera hotkey
        elif( quadrant == 7 ):
            self.inputManager.press( "f7" )
            self.detector.add_tick_action( "F7" )
            
        ## Camera hotkey
        elif( quadrant == 8 ):
            self.inputManager.press( "f8" )
            self.detector.add_tick_action( "F8" )
            
        ## Camera hotkey
        elif( quadrant == 9 ):
            self.inputManager.press( "f9" )
            self.detector.add_tick_action( "F9" )            
                
    # Detect when the cursor is inside the command area
    def detect_command_area( self ):
        return self.detector.detect_inside_minimap( 1521, 815, 396, 266 )

    # Detect when the cursor is inside the command area
    def detect_selection_tray( self ):
        return self.detector.detect_inside_minimap( 360, 865, 1000, 215 )        
                
    # Drag mouse for selection purposes
    def drag_mouse( self, should_drag ):
        if( self.should_drag != should_drag ):
            if( should_drag == True ):
                self.inputManager.mouseDown()
                self.detector.add_tick_action( "Mouse drag" )
            else:
                self.inputManager.mouseUp()
                
        self.should_drag = should_drag

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
                
            update_overlay_image( "mode-starcraft-%s" % ( "-".join( modes ) ) )
            
    def update_command_file( self, dataDicts ):
        with open(COMMAND_FILE, 'r+') as fp:
        
            # Read initial data first
            ctrl_shift_alt = fp.readline()
            sound = fp.readline().rstrip("\n")
            command = fp.readline().rstrip("\n")
            times = fp.readline().rstrip("\n")
            if (times == ""):
                times = 0                

            ctrl_shift_alt = ""
            if (self.ctrlKey):
                ctrl_shift_alt = "ctrl"
            if (self.shiftKey):
                ctrl_shift_alt = ctrl_shift_alt + "shift"
            if (self.altKey):
                ctrl_shift_alt = ctrl_shift_alt + "alt"                    
            
            if (len(self.detector.tickActions) > 1):
                sound = self.strat_to_sound(self.detector.tickActions[0])
                
                new_command = self.detector.tickActions[-1]
                if (new_command == command):
                    times = int(times) + 1
                else:
                    times = 1
                command = new_command
                fp.truncate(0)
                
            # Start writing new information
            fp.seek(0)
            fp.write(ctrl_shift_alt + '\n')
            fp.write(sound + '\n')
            fp.write(command + '\n')
            fp.write(str(times))
            fp.close()

    def strat_to_sound(self, strategy):
        sound = self.patterns[strategy]['sound']
        sound = "/" + sound.replace("general_", "").replace("vowel_", "").replace("sibilant_", "").replace("thrill_", "").replace("sound_", "").replace("fricative_", "").replace("_alveolar", "").replace("approximant_", "") + "/"
        return sound
    

    def exit( self ):
        if( self.mode == "speech" ):
            self.toggle_speech()
    
        self.release_hold_keys()    
        self.mode = "regular"
        turn_on_sound()
        update_overlay_image( "default" )
        toggle_eyetracker()
        