from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo
import time
import numpy as np
import threading

def rotateMouse( radians, radius ):
	theta = np.radians( radians )
	c, s = np.cos(theta), np.sin(theta)
	R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
	
	mousePos = np.array([radius, radius])
	relPos = np.dot( mousePos, R )
	moveTo( 500 + relPos.flat[0], 500 + relPos.flat[1] )

def printit():
  t = threading.Timer(0.1, printit)
  t.daemon = True
  t.start()
  
  radius = 20
  rotateMouse( np.abs( np.abs( time.time() * 400 ) % 360 ), radius )
  
moveTo(500, 500 )
time.sleep( 1 )
printit()


while True:
	i = 1