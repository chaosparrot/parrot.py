from lib.detection_strategies import single_tap_detection, loud_detection, medium_detection, percentage_detection
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position
from time import sleep
from subprocess import call
from lib.system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
import os

class YoutubeMode:

	def __init__(self, modeSwitcher):
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher

	def start( self ):
		turn_on_sound()
		
		moveTo( 500, 500 )
		click()
		press('space')
		press('space')		
		moveTo(500, 2000)

		press('f')
		
	def handle_input( self, dataDicts ):
		if( percentage_detection(dataDicts, "whistle", 90 ) or percentage_detection(dataDicts, "bell", 90 ) ):
			self.modeSwitcher.switchMode('browse')
			
	def exit( self ):
		self.mode = "regular"
		mute_sound()
		press('space')
		press('esc')