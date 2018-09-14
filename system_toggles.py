from subprocess import call
from pyautogui import press, hotkey, scroll, typewrite, moveRel, moveTo, position, mouseDown, mouseUp
from time import sleep

def toggle_sound():
	call(["nircmd.exe", "mutesysvolume", "2"])

def toggle_eyetracker():
	press('f4')
	
def toggle_speechrec():
	currentX, currentY = position()
	moveTo( 830, 20 )
	
	# Because windows speech rec does not allow me to turn on listening with a programmatic keypress,
	# I need to add a small timer so I can click it myself
	sleep( 1 )
	
	moveTo( currentX, currentY )