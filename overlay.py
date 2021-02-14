import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import time;
import pyautogui;
from config.config import *
import winsound
import multiprocessing
import os
import lib.ipc_manager as ipc_manager
    
def loop_overlay():
    time.sleep( 2 )
    root = tk.Tk()
    img = ImageTk.PhotoImage(Image.open(OVERLAY_FOLDER + "/default.png"))

    root.attributes('-alpha', 0.3) #For icon
    root.iconify()
    window = tk.Toplevel(root)
    window.overrideredirect(1) #Remove border
    window.attributes('-topmost', -1) # Keep on top
    window.geometry('%dx%d+%d+%d' % (40, 40, 1840, 250))
    window.attributes("-transparentcolor", "#FFFFFF") # Remove background
    
    panel = tk.Label(window, image = img)
    panel.pack(side = "bottom", fill = "both", expand = "yes")
    timestamp = time.time()

    current_overlay_status = ""
    while True:
        time.sleep( 0.016 )
        
        if (USE_COORDINATE_FILE == False):
            mouseX, mouseY = pyautogui.position()
            window.geometry("+" + str( mouseX - 20 ) + "+" + str( mouseY - 20 ))
        else:
            with open(COORDINATE_FILEPATH) as cfp:
                coords = cfp.readline().split(",")
                if (len(coords) == 2):
                    coordsX = max( 0, int(float(coords[0])))
                    coordsY = max( 0, int(float(coords[1])))
                    window.geometry("+" + str( coordsX - 20 ) + "+" + str( coordsY - 20 ))
                cfp.close()

        overlay_status = ipc_manager.getOverlayImage()
        if( overlay_status != "" and current_overlay_status != overlay_status and os.path.isfile( OVERLAY_FOLDER + "/" + overlay_status + ".png") ):
            current_overlay_status = overlay_status
            img2 = ImageTk.PhotoImage(Image.open(OVERLAY_FOLDER + "/" + current_overlay_status + ".png"))
            panel.configure(image=img2)
            panel.image = img2

        window.update_idletasks()
        window.update()

loop_overlay()