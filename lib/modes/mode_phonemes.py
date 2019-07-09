from lib.detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position, keyUp, keyDown, mouseUp, mouseDown
from time import sleep
from subprocess import call
from lib.system_toggles import toggle_eyetracker, turn_on_sound, mute_sound, toggle_speechrec
from lib.pattern_detector import PatternDetector
from lib.heroes_grammar import *
import os
import pythoncom
from lib.overlay_manipulation import update_overlay_image

class PhonemesMode:
			
	def __init__(self, modeSwitcher):
	
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher
		self.detector = PatternDetector({
			'silence': {
				'strategy': 'rapid',
				'sound': 'silence',
				'percentage': 70,
				'intensity': 0
			}
		})
		
		self.remembered_phonemes = []
		
	def start( self ):
		mute_sound()
		toggle_eyetracker()
		update_overlay_image( "default" )
			
	def handle_input( self, dataDicts ):
		self.detector.tick( dataDicts )

		# Early escape for performance
		if( self.detector.detect('silence') ):
			if( len( self.remembered_phonemes ) > 0 ):
				typewrite("->" + "/".join( self.remembered_phonemes ) + "<-" )
				self.remembered_phonemes = []
				press('enter')
		else:
			lastDict = dataDicts[ len( dataDicts ) - 1 ]
			for label in lastDict:
				if( lastDict[label]['winner'] == True and lastDict[label]['percent'] > 85 ):
					self.add_phoneme( label )
					
		return self.detector.tickActions
		
	def label_to_phoneme( self, label ):
		return label.replace( "vowel_", "" ).replace( "approximant_", "" ).replace( "fricative_", "").replace( "semivowel_", "" ).replace( "nasal_", "" ).replace( "stop_", "" ).replace( 
			"sibilant_", "" ).replace( "click_alveolar", "*").replace( "click_lateral", "^").replace( "thrill_", "~" )
		
	def add_phoneme( self, label ):
		phoneme = self.label_to_phoneme( label )
		if( len( self.remembered_phonemes ) == 0 or
			self.remembered_phonemes[ len( self.remembered_phonemes ) - 1 ] != phoneme ):
			self.remembered_phonemes.append( phoneme )
		
	def exit( self ):
		self.mode = "regular"
		turn_on_sound()
		update_overlay_image( "default" )
		toggle_eyetracker()
		