from detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position, keyUp, keyDown, mouseUp, mouseDown
from time import sleep
from subprocess import call
from system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
import os

class HeroesMode:

	def __init__(self, modeSwitcher):
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher
		self.pressed_keys = []
		self.should_follow = False

	def start( self ):
		mute_sound()
		toggle_eyetracker()
		self.camera_movement( -1 )
				
	def handle_input( self, dataDicts ):
		quadrant = detect_mouse_quadrant( 4, 3 )
	
		if( percentage_detection(dataDicts, "sound_thr", 10 ) ):
			print( "PERCENT!" )
			self.camera_movement( quadrant )
			self.mode = "movement"
		elif( self.mode == "movement"):
			print( "NOT PERCENT!" )
		
			self.camera_movement( -1 )
			self.mode = "regular"
	
		if( single_tap_detection(dataDicts, "peak_sound_haha", 70, 100 ) ):
			self.character_movement( quadrant )
			
		if( single_tap_detection(dataDicts, "peak_sound_q", 50, 1000 ) ):
			self.press_ability( 'q' )
		elif( single_tap_detection(dataDicts, "peak_sound_s", 60, 1000 ) ):
			self.press_ability( 'w' )			
		elif( single_tap_detection(dataDicts, "peak_sound_f", 40, 1000 ) ):
			self.press_ability( 'e' )
		elif( percentage_detection(dataDicts, "whistle", 70 ) ):
			self.press_ability( 'r' )
	
		if( single_tap_detection(dataDicts, "cluck", 35, 1000 ) ):
			if( self.mode == "movement" ):
				self.mode = "regular"
				
				print( "CLUCK MOVEMENT!" )				
				self.camera_movement( -1 )

			self.follow_mouse( False )
			#click(button='right')
			print( "RMB" )
		elif( single_tap_detection(dataDicts, "fingersnap", 30, 500 ) ):
			print( "LMB" )
			#click()
			
		if( percentage_detection(dataDicts, "bell", 90 ) ):
			self.modeSwitcher.switchMode('browse')
			
	def character_movement( self, quadrant ):
		print ( "Character movement!" ) 
	
		## Mount up
		if( quadrant >= 9 ):
			#press('z')
			print( "Pressing Z" )
		elif( quadrant > 4 and quadrant < 9 ):
			self.follow_mouse( True )
		## Hearth home
		elif( quadrant >= 0 and quadrant < 5 ):
			#press('b')
			print( "Pressing B" )
			
			
	def press_ability( self, key ):
		print( "pressing " + key )
		
	def camera_movement( self, quadrant ):
		print ( "Camera movement!" ) 
	
		detected = []
		self.follow_mouse( False )
		if( quadrant <= 4 and quadrant >= 0 ):
			detected.append( "up" )
		elif( quadrant >= 9 ):
			detected.append( "down" )
			
		if( quadrant == 1 or quadrant == 5 or quadrant == 9 ):
			detected.append("left")
		elif( quadrant == 4 or quadrant == 8 or quadrant == 12 ):
			detected.append("right")
		
		print( self.pressed_keys, detected )
		
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
								
		## Center area detected - Put camera to follow hero
		if( len( self.pressed_keys ) > 0 and len( detected ) == 0 ):
			press('space')
			self.camera_mode = "following"
			print ( "space!" )
		else:
			self.camera_mode = "moving"

		self.pressed_keys = detected

	def follow_mouse( self, should_follow ):
		if( self.should_follow != should_follow ):
			if( should_follow == True ):
				print( "Following mouse!" )
				#mouseDown(button="right")
			else:
				print( "Stopped following mouse!" )
				#mouseUp(button='right')
			
		self.should_follow = should_follow
		
	def exit( self ):
		self.mode = "regular"
		turn_on_sound()
		toggle_eyetracker()
