from detection_strategies import single_tap_detection, loud_detection, medium_detection
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position
import time
from subprocess import call
import os

class BrowseMode:

	def __init__(self):
		self.mode = "regular"

	def start( self ):
		self.mode = "regular"
		self.centerXPos, self.centerYPos = pyautogui.position()
		self.toggle_sound()

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
		elif( medium_detection(dataDicts, "bell", 90, 1000 ) ):
			print( 'medium!' )
			hotkey('alt', 'left')
		
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

		
	def toggle_sound( self ):
		call(["nircmd.exe", "mutesysvolume", "2"])
			
	def exit( self ):
		self.mode = "regular"
		self.toggle_sound()