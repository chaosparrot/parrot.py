from dragonfly import Grammar, CompoundRule
import pythoncom
from time import sleep
import pyautogui
from switch import *
from mode_twitch import *
from mode_browse import *
from mode_youtube import *
from system_toggles import toggle_speechrec

class ModeSwitcher(object):
	__instance = None
	__currentMode = None
	
	__modes = {}
	
	def __new__(cls):
		if ModeSwitcher.__instance is None:
			ModeSwitcher.__instance = object.__new__(cls)
			
			ModeSwitcher.__modes = {
				'browse': BrowseMode(ModeSwitcher.__instance),
				'youtube': YoutubeMode(ModeSwitcher.__instance),
				'twitch': TwitchMode(ModeSwitcher.__instance),
				'switch': SwitchMode(ModeSwitcher.__instance)
			}
			
		return ModeSwitcher.__instance
		
	def getMode(self):
		return ModeSwitcher.__currentMode
		
	def turnOnModeSwitch(self):
		self.switchMode( 'switch' )
		
	def switchMode( self, nextMode ):
		print( "Switching to " + nextMode )
		if( ModeSwitcher.__currentMode is None ):
			ModeSwitcher.__currentMode = self.__modes[nextMode]
			ModeSwitcher.__currentMode.start()
		else:
			ModeSwitcher.__currentMode.exit()
			ModeSwitcher.__currentMode = self.__modes[nextMode]
			ModeSwitcher.__currentMode.start()
			