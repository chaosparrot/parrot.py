from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position
import time
from subprocess import call
import os
from lib.system_toggles import mute_sound, toggle_speechrec, toggle_eyetracker

class BrowseMode:

	def __init__(self, modeSwitcher):
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher

	def start( self ):
		self.mode = "regular"
		self.centerXPos, self.centerYPos = pyautogui.position()
		toggle_eyetracker()
		mute_sound()

		self.fluent_mode()

	def handle_input( self, dataDicts ):
		mouseMoving = loud_detection(dataDicts, "whistle" )
		if( single_tap_detection(dataDicts, "cluck", 35, 1000 ) ):
			click()		
		elif( single_tap_detection(dataDicts, "fingersnap", 50, 1000 ) ):
			click(button='right')
		elif( mouseMoving == True ):						
			if( self.mode != "precision" ):
				if( self.mode == "regular" ):
					press("f4")
				self.mode = "precision"
				self.centerXPos, self.centerYPos = pyautogui.position()
		elif( loud_detection(dataDicts, "peak_sound_f" ) ):
			scroll( 150 )
		elif( loud_detection(dataDicts, "peak_sound_s" ) ):
			scroll( -150 )
		elif( loud_detection(dataDicts, "bell" ) ):
			self.modeSwitcher.turnOnModeSwitch()
		elif( percentage_detection(dataDicts, "sound_thr", 75 ) ):
			quadrant = detect_mouse_quadrant( 3, 3 )
			if( quadrant == 1 ):
				hotkey('alt', 'left')
			elif( quadrant == 2 ):
				hotkey('ctrl', 't')
			elif( quadrant == 3 ):
				hotkey('ctrl', 'w')
			elif( quadrant > 3 ):
				self.modeSwitcher.turnOnModeSwitch()				

			
		if( self.mode == "precision" or self.mode == "pause" ):
			if( mouseMoving == False ):
				self.mode = "pause"
			
			if( no_detection(dataDicts, "whistle") ):
				self.mode = "regular"
				press("f4")


	def fluent_mode( self ):
		t = threading.Timer(0.05, self.fluent_mode)
		t.daemon = True
		t.start()
		
		if( self.mode == "precision" ):
			self.rotateMouse( np.abs( np.abs( time.time() * 200 ) % 360 ), 20 )

	def rotateMouse( self, radians, radius ):
		theta = np.radians( radians )
		c, s = np.cos(theta), np.sin(theta)
		R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
		
		mousePos = np.array([radius, radius])
		relPos = np.dot( mousePos, R )
		moveTo( self.centerXPos + relPos.flat[0], self.centerYPos + relPos.flat[1] )

					
	def exit( self ):
		self.mode = "regular"
		toggle_eyetracker()
