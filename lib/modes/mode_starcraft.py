from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position, keyUp, keyDown, mouseUp, mouseDown
from time import sleep
from subprocess import call
from lib.system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
from lib.pattern_detector import PatternDetector
from config.config import *
import os
import pythoncom
from lib.overlay_manipulation import update_overlay_image
from lib.grammar.chat_grammar import *

class StarcraftMode:
			
	def __init__(self, modeSwitcher):
		if( SPEECHREC_ENABLED == True ):
			self.grammar = Grammar("Starcraft")
			self.chatCommandRule = ChatCommandRule()
			self.chatCommandRule.set_callback( self.toggle_speech )
			self.grammar.add_rule( self.chatCommandRule )		
	
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher
		self.detector = PatternDetector({
			'select': {
				'strategy': 'continuous',
				'sound': 'sibilant_s',
				'percentage': 60,
				'intensity': 800,
				'lowest_percentage': 40,
				'lowest_intensity': 800,
				'throttle': 0.01
			},
			'rapidclick': {
				'strategy': 'combined',
				'sound': 'sibilant_ch',
				'secondary_sound': "fricative_v",
				'percentage': 45,
				'ratio': 0,	
				'intensity': 500,
				'throttle': 0.3
			},
			'click': {
				'strategy': 'combined',
				'sound': 'click_alveolar',
				'secondary_sound': "vowel_aa",
				'percentage': 79,
				'ratio': 2,				
				'intensity': 1650,
				'throttle': 0.15
			},
			'toggle_speech': {
				'strategy': 'rapid',
				'sound': 'sound_finger_snap',
				'percentage': 90,
				'intensity': 25000,
				'throttle': 0.2
			},			
			'movement': {
				'strategy': 'frequency_threshold',
				'sound': 'sound_whistle',
				'below_frequency': 49,
				'percentage': 90,
				'intensity': 1000,
				'throttle': 0.2
			},
			'control': {
				'strategy': 'combined',
				'sound': 'vowel_oh',
				'secondary_sound': 'vowel_u',				
				'percentage': 80,
				'ratio': 1,
				'intensity': 1000,
				'throttle': 0.2
			},
			'shift': {
				'strategy': 'rapid',
				'sound': 'sibilant_sh',
				'percentage': 90,
				'frequency': 300,
				'intensity': 1600,
				'ratio': 0,
				'throttle': 0.4
			},
			'alt': {
				'strategy': 'frequency_threshold',
				'sound': 'sound_whistle',
				'percentage': 70,
				'intensity': 1000,
				'above_frequency': 50,
				'throttle': 0.3
			},
			'camera': {
				'strategy': 'combined',
				'sound': 'vowel_y',
				'secondary_sound': 'vowel_eu',	
				'percentage': 75,
				'ratio': 1.8,
				'intensity': 5000,
				'throttle': 0.2
			},
			'first_ability': {
				'strategy': 'combined',
				'sound': 'vowel_ow',
				'secondary_sound': 'vowel_u',
				'percentage': 70,
				'intensity': 1000,
				'ratio': 0.3,
				'throttle': 0.3
			},
			'second_ability': {
				'strategy': 'combined',
				'sound': 'vowel_ae',
				'secondary_sound': 'vowel_y',	
				'percentage': 75,
				'intensity': 1100,
				'ratio': 0.8,
				'throttle': 0.15
			},
			'r': {
				'strategy': 'rapid',
				'sound': 'fricative_f',
				'percentage': 90,
				'intensity': 6000,
				'throttle': 0.2
			},
			'grid_ability': {
				'strategy': 'continuous',
				'sound': 'vowel_aa',
				'percentage': 90,
				'intensity': 1000,
				'lowest_percentage': 12,
				'lowest_intensity': 500
			},
			'numbers': {
				'strategy': 'combined',
				'sound': 'vowel_iy',
				'secondary_sound': 'vowel_y',				
				'percentage': 75,
				'intensity': 2500,
				'ratio': 2,
				'throttle': 0.18
			},
			'menu': {
				'strategy': 'rapid',
				'sound': 'sound_call_bell',
				'percentage': 90,
				'intensity': 5000,
				'throttle': 0.3
			}
		})

		self.KEY_DELAY_THROTTLE = 0.5
		
		self.pressed_keys = []
		self.should_follow = False
		self.should_drag = False
		self.last_control_group = -1
		self.ability_selected = False
		self.ctrlKey = False
		self.shiftKey = False
		self.altKey = False
		self.hold_down_start_timer = 0
		
		self.hold_key = ""

	def toggle_speech( self ):
		self.release_hold_keys()
		
		if( self.mode == "regular" ):
			self.mode = "speech"
			self.grammar.load()
			press('enter')
		else:
			self.mode = "regular"
			self.grammar.unload()
		toggle_speechrec()
		
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
				keyDown('shift')
				self.shiftKey = shift			
				print( 'Holding SHIFT' )
				self.update_overlay()				
			else:
				keyUp('shift')
				self.shiftKey = shift
				print( 'Releasing SHIFT' )				
				self.update_overlay()
	
	def hold_alt( self, alt ):
		if( self.altKey != alt ):
			if( alt == True ):
				self.altKey = alt			
				print( 'Enabling ALT' )
				self.update_overlay()
				self.detector.set_throttle( 'first_ability', 0.1 )
				self.detector.set_throttle( 'second_ability', 0.1 )
			else:
				self.altKey = alt
				print( 'Disabling ALT' )
				self.update_overlay()
				self.detector.set_throttle( 'first_ability', 0.3 )
				self.detector.set_throttle( 'second_ability', 0.15 )
	
	def hold_control( self, ctrlKey ):
		if( self.ctrlKey != ctrlKey ):
			if( ctrlKey == True ):
				keyDown('ctrl')
				print( 'Holding CTRL' )
				self.ctrlKey = ctrlKey
				self.update_overlay()				
			else:
				keyUp('ctrl')
				print( 'Releasing CTRL' )
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
		if( self.detector.detect("toggle_speech" ) ):
			quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
			
			if( quadrant3x3 < 6 ):
				self.release_hold_keys()
				self.toggle_speech()
				
			return self.detector.tickActions			
		# Recognize speech commands in speech mode
		elif( self.mode == "speech" ):
			pythoncom.PumpWaitingMessages()
			
			return self.detector.tickActions			
			
		# Regular quick command mode
		elif( self.mode == "regular" ):
			self.handle_quick_commands( dataDicts )
			
		return self.detector.tickActions
	
	def handle_quick_commands( self, dataDicts ):
		# Early escape for performance
		if( self.detector.detect_silence() ):
			self.drag_mouse( False )
			self.hold_down_start_timer = 0
			return
			
		# Selecting units
		selecting = not self.detector.is_throttled('rapidclick') and self.detector.detect( "select" )
		if( self.ability_selected and selecting ):
			click(button='left')
			print( "LMB" )
			self.ability_selected = False
			
			# Clear the throttles for abilities
			self.detector.clear_throttle('camera')
			self.detector.clear_throttle('first_ability')
			self.detector.clear_throttle('second_ability')
			
			self.detector.throttle('rapidclick')
		else:
			self.drag_mouse( selecting )
		
		## Press Grid ability
		if( self.detector.detect("grid_ability") ):
			quadrant4x3 = self.detector.detect_mouse_quadrant( 4, 3 )
			if( time.time() - self.hold_down_start_timer > self.KEY_DELAY_THROTTLE ):
				self.use_ability( quadrant4x3 )
				self.release_hold_keys()
				self.hold_shift( False )
			
			if( self.hold_down_start_timer == 0 ):
				self.hold_down_start_timer = time.time()
		else:
			self.hold_down_start_timer = 0
		
		if( selecting ):
			self.ability_selected = False
			self.hold_control( False )

		elif( self.detector.detect( "click" ) ):
		
			# Cast selected ability or Ctrl+click
			if( self.detect_command_area() or self.ability_selected == True or self.ctrlKey == True or self.altKey == True or ( self.shiftKey and self.detect_selection_tray() ) ):
				click(button='left')
				print( "LMB" )
			else:
				click(button='right')
				print( "RMB" )
				
			# Release the held keys - except when shift clicking units in the selection tray ( for easy removing from the unit group )
			if( not( self.shiftKey and self.detect_selection_tray() ) ):
				self.release_hold_keys()
		elif( ( self.detector.is_throttled('camera') or self.detector.is_throttled("first_ability") or self.detector.is_throttled('second_ability') ) and self.detector.detect( "rapidclick" ) ):
			click(button='left')
			self.ability_selected = False
			
			# Clear the throttles for abilities
			self.detector.clear_throttle('camera')
			self.detector.clear_throttle('first_ability')
			self.detector.clear_throttle('second_ability')
			print( "LMB" )
			
		# CTRL KEY holding
		elif( self.detector.detect( "control" ) ):
			self.hold_control( True )
			
		# SHIFT KEY holding / toggling
		elif( self.detector.detect( "shift" ) ):
			self.hold_shift( not self.shiftKey )
	
		# ALT KEY holding / toggling
		elif( self.detector.detect( "alt" ) ):
			self.hold_alt( not self.altKey )

		## Movement options
		elif( self.detector.detect( "movement" ) ):
			quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
			if( quadrant3x3 <= 5 or quadrant3x3 == 7 ):
				self.cast_ability( 'a' )
			elif( quadrant3x3 == 6 ):
				self.press_ability( 'h' )
			elif( quadrant3x3 == 8 ):
				self.press_ability( 's' )
			elif( quadrant3x3 == 9 ):
				self.cast_ability( 'p' )
				
			self.hold_shift( False )				
		## Press Q
		elif( self.detector.detect( "first_ability" ) ):
			self.ability_selected = True
			self.detector.clear_throttle('rapidclick')
			
			if( self.altKey ):
				print( "pressing Z" )
				press( 'z' )
				
			else:
				print( "pressing Q" )
				press( 'q' )
			
		## Press W
		elif( self.detector.detect( "second_ability") ):
			self.ability_selected = True
			self.detector.clear_throttle('rapidclick')		
		
			if( self.altKey ):
				print( "pressing X" )
				press( 'x' )
			else:
				print( "pressing W" )
				press( 'w' )
				
		## Press R ( Burrow )
		elif( self.detector.detect( "r") ):
		
			if( self.altKey ):
				print( "pressing C" )
				press( 'c' )
			else:
				print( "pressing R" )
				press( 'r' )
		
		elif( self.detector.detect( "menu" ) ):
			self.release_hold_keys()
			
			quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
			if( quadrant3x3 == 9 ):
				self.press_ability( 'f10' )
			else:
				self.press_ability( 'esc' )
			
		## Move the camera
		elif( not self.detector.is_throttled('second_ability') and self.detector.detect( "camera" ) ):
			quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
			self.camera_movement( quadrant3x3 )
			self.hold_shift( False )
			self.hold_alt( False )
						
		## Press control group ( only allow CTRL and SHIFT )
		elif( self.detector.detect( "numbers" ) ):
			if( self.altKey ):
				keyDown('alt')
		
			quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
			self.use_control_group( quadrant3x3 )
			
			if( self.altKey ):
				keyUp('alt')
				
			self.hold_alt( False )				
			self.hold_control( False )
			self.hold_shift( False )

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
		
	def press_ability( self, key ):
		print( "pressing " + key )
		press( key )
		self.release_hold_keys()
		
	def camera_movement( self, quadrant ):
		## Move camera to kerrigan when looking above the UI
		if( quadrant < 3 ):
			self.press_ability( "f1" )
		elif( quadrant == 2 ):			
			press( "f2" )
		elif( quadrant > 3 and quadrant < 7 ):
			press( "backspace" )
			self.detector.clear_throttle('rapidclick')
		
		## Move camera to danger when when looking at the minimap or unit selection
		elif( quadrant == 7 or quadrant == 8 ):
			self.press_ability( "space" )
			
		## Follow the unit when looking near the command card
		elif( quadrant == 9 ):
			self.press_ability( "." )
				
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
				mouseDown()
			else:
				mouseUp()
				
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

	def exit( self ):
		if( self.mode == "speech" ):
			self.toggle_speech()
	
		self.release_hold_keys()	
		self.mode = "regular"
		turn_on_sound()
		update_overlay_image( "default" )
		toggle_eyetracker()
		