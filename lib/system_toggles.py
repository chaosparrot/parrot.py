from config.config import *
from subprocess import call
from pyautogui import press, hotkey, scroll, typewrite, moveRel, moveTo, position, mouseDown, mouseUp
from time import sleep
import os  

def toggle_sound():
	if( os.path.isfile('./nircmd.exe') ):
		call(["nircmd.exe", "mutesysvolume", "2"])
	
def mute_sound():
	if( os.path.isfile('./nircmd.exe') ):
		print("")
		#call(["nircmd.exe", "mutesysvolume", "1"])
	
def turn_on_sound():
	if( os.path.isfile('./nircmd.exe') ):
		print("")
		#call(["nircmd.exe", "mutesysvolume", "0"])

def toggle_eyetracker():
	press(EYETRACKING_TOGGLE)
	
def toggle_speechrec():
	if( SPEECHREC_ENABLED == True ):
		hotkey('ctrl','win')
		print( "TOGGLE!" )