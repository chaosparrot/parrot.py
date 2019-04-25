from config.config import *
from subprocess import call
from pyautogui import press, hotkey, scroll, typewrite, moveRel, moveTo, position, mouseDown, mouseUp
from time import sleep

def toggle_sound():
	call(["nircmd.exe", "mutesysvolume", "2"])
	
def mute_sound():
	print("")
	#call(["nircmd.exe", "mutesysvolume", "1"])
	
def turn_on_sound():
	print("")
	#call(["nircmd.exe", "mutesysvolume", "0"])

def toggle_eyetracker():
	press('f4')
	
def toggle_speechrec():
	if( SPEECHREC_ENABLED == True ):
		hotkey('ctrl','win')