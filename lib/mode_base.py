from lib.detection_strategies import *
from config.config import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position, keyUp, keyDown, mouseUp, mouseDown
from time import sleep
from subprocess import call
from lib.system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
from lib.pattern_detector import PatternDetector
import os

if( SPEECHREC_ENABLED == True ):
	import pythoncom

class BaseMode:
	
	modes = []
		
	def __init__(self, modeSwitcher, config):
		self.modeSwitcher = modeSwitcher
		self.detector = new PatternDetector( config )
		if( SPEECHREC_ENABLED ):
			self.grammar = new Grammar()

	def start( self ):
		mute_sound()
		toggle_eyetracker()
				
	def handle_input( self, dataDicts ):
		self.detector.tick( dataDicts )
	
		return self.detector.tickActions

	def load_grammar( self ):
		self.grammar.load()
		self.modes.append( 'speech' )
		
	def unload_grammar( self ):
		self.grammar.unload()
		
	def run( self ):
		print( "DEFAULT RUN!" )
		
	def exit( self ):
		turn_on_sound()
		toggle_eyetracker()
		