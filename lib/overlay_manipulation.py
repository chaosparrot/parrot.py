from config.config import *
import os
import multiprocessing
if( OVERLAY_ENABLED ):
	import tkinter as tk
	from tkinter import *
	from PIL import ImageTk, Image
	import pyautogui;

## Updates the overlay image - But checks if the image exists first to prevent
## Unexpected crashes from occuring during the overlay phase
def update_overlay_image( overlay_mode ):
	if( OVERLAY_ENABLED ):
		filehandle = open(OVERLAY_FILE, 'w')
		if( os.path.isfile( OVERLAY_FOLDER + "/" + overlay_mode + ".png") ):  
			filehandle.write(overlay_mode)  
		else:
			filehandle.write("default")
		filehandle.close()
		
def run_overlay():
	root = tk.Tk()
	img = ImageTk.PhotoImage(Image.open(OVERLAY_FOLDER + "/default.png" ))

	root.attributes('-alpha', 0.3) #For icon
	root.iconify()
	window = tk.Toplevel(root)
	window.overrideredirect(1) #Remove border
	window.attributes('-topmost', -1) # Keep on top
	window.geometry('%dx%d+%d+%d' % (20, 20, 1840, 250))
	window.attributes("-transparentcolor", "#FFFFFF") # Remove background

	panel = tk.Label(window, image = img)
	panel.pack(side = "bottom", fill = "both", expand = "yes")
	timestamp = time.time()

	current_overlay_status = ""
	while True:
		mouseX, mouseY = pyautogui.position()
		window.geometry("+" + str( mouseX + 40 ) + "+" + str( mouseY + 40 ))
		filepath = OVERLAY_FILE
		
		with open(filepath) as fp: 
			print( "asdf " )
			overlay_status = fp.readline()
			
			if( overlay_status != "" and current_overlay_status != overlay_status ):
				current_overlay_status = overlay_statusq
				img2 = ImageTk.PhotoImage(Image.open(OVERLAY_FOLDER + "/" + current_overlay_status + ".png"))
				panel.configure(image=img2)
				panel.image = img2
			fp.close()

		window.update_idletasks()
		window.update()