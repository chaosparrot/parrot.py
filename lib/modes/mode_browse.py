from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position
import time
from subprocess import call
import os
from lib.system_toggles import mute_sound, toggle_speechrec, toggle_eyetracker
from lib.pattern_detector import PatternDetector

class BrowseMode:

	def __init__(self, modeSwitcher):
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher
		self.detector = PatternDetector({
			'click': {
				'strategy': 'rapid',
				'sound': 'cluck',
				'percentage': 50,
				'intensity': 600,
				'throttle': 0.05
			},
			'rightclick': {
				'strategy': 'single_tap',
				'sound': 'fingersnap',
				'percentage': 50,
				'intensity': 1000
			},
			'moving': {
				'strategy': 'continuous',
				'sound': 'whistle',
				'percentage': 70,
				'intensity': 500,
				'lowest_percentage': 30,
				'lowest_intensity': 500
			},
			'scroll_up': {
				'strategy': 'rapid',
				'sound': 'sound_f',
				'percentage': 70,
				'intensity': 1000
			},
			'scroll_down': {
				'strategy': 'rapid',
				'sound': 'sound_s',
				'percentage': 70,
				'intensity': 1000
			},
			'special': {
				'strategy': 'rapid',
				'sound': 'sound_uuh',
				'percentage': 70,
				'intensity': 1000,
				'throttle': 0.3
			},
			'exit': {
				'strategy': 'rapid',
				'sound': 'hotel_bell',
				'percentage': 90,
				'intensity': 1000
			}
		})

	def start( self ):
		self.mode = "regular"
		self.centerXPos, self.centerYPos = pyautogui.position()
		toggle_eyetracker()
		mute_sound()

		self.fluent_mode()

	def handle_input( self, dataDicts ):
		self.detector.tick( dataDicts )
		
		# Early return for quicker detection
		if( self.detector.detect_silence() ):
			return self.detector.tickActions
				
		mouseMoving = False
		if( self.detector.detect("click") ):
			click()
		elif( self.detector.detect("moving" ) ):
			mouseMoving = True
			if( self.mode != "precision" ):
				if( self.mode == "regular" ):
					press("f4")
				self.mode = "precision"
				self.centerXPos, self.centerYPos = pyautogui.position()
		elif( self.detector.detect("rightclick") ):
			click(button='right')
		elif( self.detector.detect("scroll_up") ):
			scroll( 150 )
		elif( self.detector.detect("scroll_down") ):
			scroll( -150 )
		elif( self.detector.detect( "exit" ) ):
			self.modeSwitcher.turnOnModeSwitch()
		elif( self.detector.detect( "special" ) ):
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

		return self.detector.tickActions

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
