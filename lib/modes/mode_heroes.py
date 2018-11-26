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
				'throttle': 0.1
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
				'strategy': 'rapid',
				'sound': 'sound_uuh',
				'percentage': 70,
				'intensity': 1000
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
		elif( self.detector.detect( "exit" ) ):
			self.modeSwitcher.switchMode('browse')
			
		return self.detector.tickActions
			
	def character_movement( self, quadrant ):
		print ( "Character movement!" ) 
		
		## Show tab
		if( quadrant == 1 ):
			self.follow_mouse( True )
		elif( quadrant == 3 ):
			self.press_ability( 'z' )
			self.follow_mouse( False )
		## Hearth home
		elif( quadrant == 7 ):
			self.press_ability('b')
		else:
			self.press_ability( 'a' )
			self.follow_mouse( False )
		
	def set_hold_key( self, quadrant ):
		if( quadrant == 1 ):
			self.hold_key = "1"
		elif( quadrant == 2 ):
			self.hold_key = "2"
		elif( quadrant == 3 ):
			self.hold_key = "3"
		elif( quadrant == 4 ):
			self.hold_key = "d"
		elif( quadrant == 6 ):
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
		
		if( quadrant == 12 ):
			self.drag_mouse( True )
		else:
			self.drag_mouse( False )		
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
