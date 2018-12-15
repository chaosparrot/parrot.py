from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position, keyUp, keyDown, mouseUp, mouseDown
from time import sleep
from subprocess import call
from lib.system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
from lib.pattern_detector import PatternDetector
import os

class HeroesMode:

	minimaps = {
		'default': [1600, 800, 300, 250],
		'warhead': [1532, 712, 400, 300],
		'braxis': [1570, 750, 250, 200],		
		'towers': [1600, 789, 300, 300]
	}
	
	heroes = {
		'default': {'aim': {'1': False, '2': False, '3': False, 'd': True, 'f': True, 'z': False}},
		'Kerrigan': {'aim': {'1': False, '2': False, '3': False, 'd': False, 'f': True, 'z': False}},
		'Dehaka': {'aim': {'1': False, '2': False, '3': False, 'd': False, 'f': True, 'z': True}}		
	}
	
	current_map = 'braxis'
	current_hero = 'default'
		
	def __init__(self, modeSwitcher):
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher
		self.detector = PatternDetector({
			'click': {
				'strategy': 'rapid',
				'sound': 'cluck',
				'percentage': 45,
				'intensity': 600,
				'throttle': 0.1
			},
			'rightclick': {
				'strategy': 'rapid',
				'sound': 'fingersnap',
				'percentage': 40,
				'intensity': 3000,
				'throttle': 0.5
			},
			'q': {
				'strategy': 'rapid',
				'sound': 'sound_oh',
				'percentage': 50,
				'intensity': 500,
				'throttle': 0.05
			},
			'w': {
				'strategy': 'rapid',
				'sound': 'sound_s',
				'percentage': 80,
				'intensity': 500,
				'throttle': 0.1
			},
			'e': {
				'strategy': 'rapid',
				'sound': 'sound_f',
				'percentage': 60,
				'intensity': 500,
				'throttle': 0.15
			},
			'heroic': {
				'strategy': 'rapid',
				'sound': 'hotel_bell',
				'percentage': 60,
				'intensity': 10000,
				'throttle': 0.3
			},
			'movement': {
				'strategy': 'rapid',
				'sound': 'sound_ooh',
				'percentage': 35,
				'intensity': 2000,
				'throttle': 0.2
			},
			'camera': {
				'strategy': 'continuous',
				'sound': 'sound_uuh',
				'percentage': 50,
				'intensity': 1000,
				'lowest_percentage': 15,
				'lowest_intensity': 400
			},
			'special': {
				'strategy': 'rapid',
				'sound': 'whistle',
				'percentage': 50,
				'intensity': 1000,
				'throttle': 0.3
			},
			'exit': {
				'strategy': 'rapid',
				'sound': 'bell',
				'percentage': 75,
				'intensity': 1000
			}
		})

		self.pressed_keys = []
		self.should_follow = False
		self.should_drag = False
		
		self.hold_key = ""

	def start( self ):
		mute_sound()
		toggle_eyetracker()
		
		
				
	def handle_input( self, dataDicts ):
		self.detector.tick( dataDicts )
	
		if( self.detector.detect( "click" ) ):
			if( self.hold_key == ""):
				self.follow_mouse( False )
				click(button='right')
				print( "RMB" )
			elif( self.hold_key == "click"):
				click()
			else:
				self.press_ability( self.hold_key )
				
			self.hold_key = ""
		elif( self.detector.detect( "q" ) ):
			self.press_ability( 'q' )
		elif( self.detector.detect( "w" ) ):
			self.press_ability( 'w' )			
		elif( self.detector.detect( "e" ) ):
			self.press_ability( 'e' )
		elif( self.detector.detect( "heroic" ) ):
			self.press_ability( 'r' )
		elif( self.detector.detect( "special" ) ):
			quadrant = self.detector.detect_mouse_quadrant( 3, 3 )
			self.set_hold_key( quadrant )
		elif( self.detector.detect( "movement" ) ):
			if( self.mode == "minimap" ):
				self.click_on_minimap()
			else:	
				quadrant = self.detector.detect_mouse_quadrant( 3, 3 )
				self.character_movement( quadrant )

		if( self.detector.detect( "camera" ) ):
			edges = self.detector.detect_mouse_screen_edge( 200 )

			self.mode = "cameramovement"
			print ( "Camera movement!" ) 
			self.camera_movement( edges, detect_mouse_quadrant( 4, 3 ) )
		elif( self.mode == "cameramovement" ):
			self.camera_movement( [], -1 )
			print( "Regular mode!" )
			self.mode = "regular"
					
		
		if( self.detector.detect( "rightclick" ) ):
			print( "LMB" )
			click()
		
		if( self.detector.detect( "exit" ) ):
			self.modeSwitcher.switchMode('browse')
			
		return self.detector.tickActions
			
	def character_movement( self, quadrant ):
		print ( "Character movement!" ) 
		
		## Show tab
		if( quadrant == 1 ):
			self.follow_mouse( True )
		elif( quadrant == 3 ):
			if( self.heroes[ self.current_hero ]['aim']['z'] == False ):
				self.press_ability("z")
			else:
				self.hold_key = "z"
			self.follow_mouse( False )
		## Hearth home
		elif( quadrant == 7 ):
			self.press_ability('b')
		else:
			self.press_ability( 'a' )
			self.follow_mouse( False )
		
	def set_hold_key( self, quadrant ):
		if( quadrant == 1 ):
			if( self.heroes[ self.current_hero ]['aim']['1'] == False ):
				self.press_ability("1")
			else:
				self.hold_key = "1"
				
		elif( quadrant == 2 ):
			if( self.heroes[ self.current_hero ]['aim']['2'] == False ):
				self.press_ability("2")
			else:
				self.hold_key = "2"
		elif( quadrant == 3 ):
			if( self.heroes[ self.current_hero ]['aim']['3'] == False ):
				self.press_ability("3")
			else:
				self.hold_key = "3"
		elif( quadrant == 4 ):
			if( self.heroes[ self.current_hero ]['aim']['d'] == False ):
				self.press_ability("d")
			else:
				self.hold_key = "d"
		elif( quadrant == 6 ):
			self.hold_key = "f"
			if( self.heroes[ self.current_hero ]['aim']['f'] == False ):
				self.press_ability("f")
			else:
				self.hold_key = "f"
		elif( quadrant == 7 ):
			self.press_ability("n")
			self.hold_key = "click"
		elif( quadrant == 8 ):
			self.press_ability( "tab" )						
		elif( quadrant == 9 ):
			self.press_ability( "f10" )						

	def press_ability( self, key ):
		print( "pressing " + key )
		press( key )
		
	def camera_movement( self, edges, quadrant ):	
		self.follow_mouse( False )
		detected = edges
		
		if( self.mode != "minimap" and quadrant == 12 ):
			self.mode = "minimap"
		else:
			# Release all the keys that it doesnt have anymore
			for held in self.pressed_keys:
				if( held not in detected ):
					keyUp( held )
					print ( "releasing " + held )
			
			## Hold down new keys
			for pressed in detected:
				if( pressed not in self.pressed_keys ):
					keyDown( pressed )
					print ( "holding down " + pressed )
					
			if( quadrant == 6 or quadrant == 7 ):
				press('space')

		self.pressed_keys = detected
		
	def click_on_minimap( self ):
		minimap = self.minimaps[ self.current_map ]
		minimapX, minimapY = self.detector.detect_minimap_position( minimap[0], minimap[1], minimap[2], minimap[3] )
		
		toggle_eyetracker()
		moveTo( minimapX, minimapY )
		click()
		toggle_eyetracker()
		self.mode = "regular"

	def drag_mouse( self, should_drag ):
		if( self.should_drag != should_drag ):
			if( should_drag == True ):
				print( "Start dragging!" )			
				mouseDown()
			else:
				print( "Stopped dragging!" )			
				mouseUp()
				
		self.should_drag = should_drag
		
	def follow_mouse( self, should_follow ):
		if( self.should_follow != should_follow ):
			if( should_follow == True ):
				print( "Following mouse!" )
				mouseDown(button="right")
			else:
				print( "Stopped following mouse!" )
				mouseUp(button='right')
			
		self.should_follow = should_follow
		
	def exit( self ):
		self.mode = "regular"
		turn_on_sound()
		toggle_eyetracker()
