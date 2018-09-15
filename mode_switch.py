from dragonfly import Grammar, CompoundRule
import pythoncom
from time import sleep
import pyautogui
from mode_twitch import *
from mode_browse import *
from mode_youtube import *
from system_toggles import toggle_speechrec

class ModeSwitcher(object):
	__instance = None
	__currentMode = None
	
	def __new__(cls):
		if ModeSwitcher.__instance is None:
			ModeSwitcher.__instance = object.__new__(cls)
		return ModeSwitcher.__instance
		
	def getMode(self):
		return ModeSwitcher.__currentMode
		
	def switchMode( self, nextMode ):
		print( nextMode )
		if( ModeSwitcher.__currentMode is None ):
			ModeSwitcher.__currentMode = nextMode
			nextMode.start()
		else:
			ModeSwitcher.__currentMode.exit()
			ModeSwitcher.__currentMode = nextMode
			ModeSwitcher.__currentMode.start()
			