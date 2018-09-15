from subprocess import call
from pyautogui import press, hotkey, scroll, typewrite, moveRel, moveTo, position, mouseDown, mouseUp
from time import sleep

def toggle_sound():
	call(["nircmd.exe", "mutesysvolume", "2"])
	
def mute_sound():
	call(["nircmd.exe", "mutesysvolume", "1"])
	
def turn_on_sound():
	call(["nircmd.exe", "mutesysvolume", "0"])

def toggle_eyetracker():
	press('f4')
	
def toggle_speechrec():
	currentX, currentY = position()
	press('f6')
	moveTo( 830, 20, 0.2 )

	# Because windows speech rec does not allow me to turn on listening with a programmatic keypress,
	# We do it using the iris program which simulates a keypress after a while
	sleep( 0.7 )
	press('f6')
		
	moveTo( currentX, currentY )