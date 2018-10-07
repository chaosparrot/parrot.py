from detection_strategies import *
import threading
import numpy as np
import pyautogui
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position
import time
from subprocess import call
import os
from system_toggles import mute_sound, toggle_speechrec, toggle_eyetracker
from dragonfly import Grammar
from excel_grammar import *
import pythoncom

class ExcelMode:

	def __init__(self, modeSwitcher, file):
		self.mode = "regular"
		self.modeSwitcher = modeSwitcher
		self.file = file
		
		self.grammar = Grammar("Excel")
		columnNumberRule = ColumnNumberPrintRule()
		self.grammar.add_rule( columnNumberRule )
		columnModeRule = ColumnModePrintRule()
		self.grammar.add_rule( columnModeRule )		
		correctionRule = CorrectionRule()
		self.grammar.add_rule( correctionRule )
		copyRowRule = CopyRowRule()
		self.grammar.add_rule( copyRowRule )
		nextRowRule = NextRowRule()
		self.grammar.add_rule( nextRowRule )

	def start( self ):
		self.grammar.load()
		self.mode = "speech"
		toggle_speechrec()
		self.centerXPos, self.centerYPos = pyautogui.position()
		toggle_eyetracker()
		mute_sound()
		self.open_file()

	def handle_input( self, dataDicts ):
		if( loud_detection(dataDicts, "bell" ) ):
			self.modeSwitcher.turnOnModeSwitch()
	
		if( self.mode == "regular" ):
			if( percentage_detection(dataDicts, "whistle", 75 ) ):
				self.mode = "speech"
				toggle_speechrec()
			else:
				if( single_tap_detection(dataDicts, "cluck", 35, 1000 ) ):
					click()		
				elif( single_tap_detection(dataDicts, "fingersnap", 50, 1000 ) ):
					click(button='right')
				elif( loud_detection(dataDicts, "peak_sound_f" ) ):
					scroll( 150 )
				elif( loud_detection(dataDicts, "peak_sound_s" ) ):
					scroll( -150 )

				if( percentage_detection(dataDicts, "sound_thr", 75 ) ):
					quadrant = detect_mouse_quadrant( 3, 3 )
					if( quadrant > 3 ):
						self.modeSwitcher.switchMode('browse')
		
		elif( self.mode == "speech" ):
			self.speech_mode()
			if( percentage_detection(dataDicts, "sound_thr", 75 ) ):
				self.mode = "regular"
				toggle_speechrec()
				
	def speech_mode(self):
		pythoncom.PumpWaitingMessages()
		time.sleep(.1)

	def open_file( self ):
		if( self.file != '' ):
			call(["start", self.file], shell=True)
					
	def exit( self ):
		if( self.mode == "speech" ):
			toggle_speechrec()
		self.mode = "regular"
		toggle_eyetracker()
		self.grammar.unload()

