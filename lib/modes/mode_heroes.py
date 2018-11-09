from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position, keyUp, keyDown, mouseUp, mouseDown
from time import sleep
from subprocess import call
from lib.system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
import os

class HeroesMode:

	def __init__(self, modeSwitcher):
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher
		self.pressed_keys = []
		self.should_follow = False
		self.should_drag = False
		
		self.hold_key = ""

	def start( self ):
		mute_sound()
		toggle_eyetracker()
				
	def handle_input( self, dataDicts ):
		if( single_tap_detection(dataDicts, "sound_ie", 70, 1000 ) ):
			quadrant = detect_mouse_quadrant( 3, 3 )
			self.character_movement( quadrant )
		elif( single_tap_detection(dataDicts, "sound_huu", 50, 1000 ) ):
			quadrant = detect_mouse_quadrant( 3, 3 )
			self.set_hold_key( quadrant )
		elif( single_tap_detection(dataDicts, "sound_oh", 50, 1000 ) ):
			self.press_ability( 'q' )
		elif( single_tap_detection(dataDicts, "sound_s", 60, 1000 ) ):
			self.press_ability( 'w' )			
		elif( single_tap_detection(dataDicts, "sound_f", 40, 1000 ) ):
			self.press_ability( 'e' )
		elif( percentage_detection(dataDicts, "whistle", 70 ) ):
			self.press_ability( 'r' )
		elif( single_tap_detection(dataDicts, "sound_pfft", 70, 1000 ) ):
			self.press_ability( 'z' )
		elif( single_tap_detection(dataDicts, "sound_tschk", 50, 1000 ) ):
			self.press_ability( 'a' )

		if( percentage_detection(dataDicts, "sound_thr", 40 ) ):
			edges = detect_screen_edge( 200 )

			self.mode = "cameramovement"
			print ( "Camera movement!" ) 
			self.camera_movement( edges, detect_mouse_quadrant( 4, 3 ) )
		elif( self.mode == "cameramovement" ):
			self.camera_movement( [], -1 )
			print( "Regular mode!" )
			self.mode = "regular"
					
		if( single_tap_detection(dataDicts, "cluck", 50, 1000 ) ):
			if( self.hold_key == ""):
				self.follow_mouse( False )
				click(button='right')
				print( "RMB" )
			elif( self.hold_key == "click"):
				click()
			else:
				self.press_ability( self.hold_key )
				
			self.hold_key = ""
		elif( single_tap_detection(dataDicts, "fingersnap", 60, 1200 ) ):
			print( "LMB" )
			click()
			
		if( percentage_detection(dataDicts, "bell", 90 ) ):
			self.modeSwitcher.switchMode('browse')
			
	def character_movement( self, quadrant ):
		print ( "Character movement!" ) 
		
		## Show tab
		if( quadrant == 1 ):
			self.hold_key = "d"
			self.follow_mouse( False )			
		elif( quadrant == 3 ):
			self.press_ability('h')
			self.follow_mouse( False )
		## Hearth home
		elif( quadrant == 7 ):
			self.press_ability('b')
		elif( quadrant == 5 ):
			self.follow_mouse( True )
		
	def set_hold_key( self, quadrant ):
		if( quadrant == 1 ):
			self.hold_key = "1"
		elif( quadrant == 2 ):
			self.hold_key = "2"
		elif( quadrant == 3 ):
			self.hold_key = "3"
		elif( quadrant == 7 ):
			self.press_ability("n")
			self.hold_key = "click"
		elif( quadrant == 9 ):
			self.press_ability( "tab" )
			
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
