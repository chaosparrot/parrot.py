from detection_strategies import single_tap_detection, loud_detection, medium_detection
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position
from time import sleep
from subprocess import call
from system_toggles import toggle_eyetracker, turn_on_sound, mute_sound
import os

class TwitchMode:

	def __init__(self):
		self.mode = "regular"

	def start( self ):
		turn_on_sound()
		toggle_eyetracker()
		
		moveTo( 500, 500 )
		click()
		moveTo(2000, 2000)
		
		hotkey('ctrl', 'f')
		
	def handle_input( self, dataDicts ):
		if( loud_detection(dataDicts, "bell", 90, 1000 ) ):
			self.exit()
					
	def exit( self ):
		self.mode = "regular"
		toggle_eyetracker()
		mute_sound()
		press('space')
		press('esc')
		