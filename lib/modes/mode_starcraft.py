from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position, keyUp, keyDown, mouseUp, mouseDown
from time import sleep
from subprocess import call
from lib.system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
from lib.pattern_detector import PatternDetector
from lib.heroes_grammar import *
import os
import pythoncom

class StarcraftMode:
			
	def __init__(self, modeSwitcher):
	
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher
		self.detector = PatternDetector({
			'select': {
				'strategy': 'rapid',
				'sound': 'sound_s',
				'percentage': 90,
				'intensity': 600,
				'throttle': 0.01
			},
			'rightclick': {
				'strategy': 'rapid',
				'sound': 'cluck',
				'percentage': 50,
				'intensity': 1000,
				'throttle': 0.1
			},
			'camera': {
				'strategy': 'continuous',
				'sound': 'sound_uuh',
				'percentage': 50,
				'intensity': 1000,
				'lowest_percentage': 15,
				'lowest_intensity': 400
			},
			'control_group': {
				'strategy': 'rapid',
				'sound': 'sound_ooh',
				'percentage': 35,
				'intensity': 1500,
				'throttle': 0.3
			},
			'building': {
				'strategy': 'rapid',
				'sound': 'whistle',
				'percentage': 50,
				'intensity': 1000,
				'throttle': 0.1
			}
		})

		self.pressed_keys = []
		self.should_follow = False
		self.should_drag = False
		self.last_control_group = -1
		
		self.hold_key = ""

	def start( self ):
		mute_sound()
		toggle_eyetracker()
				
	def handle_input( self, dataDicts ):
		self.detector.tick( dataDicts )
		
		## Select units
		self.drag_mouse( self.detector.detect( "select" ) )
			
		## Detect mouse position
		quadrant3x3 = self.detector.detect_mouse_quadrant( 3, 3 )
			
		## Movement
		if( self.detector.detect( "rightclick" ) ):
			click(button='right')
			
		## Grid detections	
		if( self.detector.detect( "camera" ) ):
			self.camera_movement( quadrant3x3 )
		elif( self.detector.detect( "building" ) ):
			quadrant4x3 = self.detector.detect_mouse_quadrant( 4, 3 )
			self.use_ability( quadrant4x3 )
		elif( self.detector.detect( "control_group" ) ):
			self.use_control_group( quadrant3x3 )
			
		return self.detector.tickActions
		
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
		
	def camera_movement( self, quadrant ):	
					
		## Move camera to danger
		if( quadrant == 7 ):
			self.press_ability( "space" )
		
		## Move back through bases		
		elif( quadrant == 8 ):
			self.press_ability( "backspace" )
			
		## Open the menu
		elif( quadrant == 9 ):
			self.press_ability( "f10" )
		
		# Attack move
		else:
			self.press_ability('a')
				
	# Drag mouse for seleciton purposes
	def drag_mouse( self, should_drag ):
		if( self.should_drag != should_drag ):
			if( should_drag == True ):
				mouseDown()
			else:
				mouseUp()
				
		self.should_drag = should_drag

	def exit( self ):
		self.mode = "regular"
		turn_on_sound()
		toggle_eyetracker()
		