import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import time;
import pyautogui;
from config.config import *
import winsound
import multiprocessing
    
def loop_overlay():
    time.sleep( 2 )
    root = tk.Tk()

    root.attributes('-alpha', 0.3) #For icon
    root.iconify()
    window = tk.Toplevel(root)
    window.configure(bg="#000000")
    ##window.overrideredirect(1) #Remove border
    ##window.attributes('-topmost', -1) # Keep on top
    ##window.geometry('%dx%d+%d+%d' % (300, 300, 0, 0))
    ##window.attributes("-transparentcolor", "#FFFFFF") # Remove background
    timestamp = time.time()
    
    text = tk.Text(window, height=1, width=20, relief="flat")
    text.configure(bg="#000000", fg="#FFFFFF", padx=20, pady=20, font=("Arial Black", 40, "bold"))
    text.insert(INSERT, "vowel_ah")
    text.place(x=60, y=0)
    
    commandText = tk.Text(window, relief="flat", width=20, height=1)
    commandText.configure(bg="#000000", fg="#FFFFFF", font=("Arial Black", 120, "bold"))
    commandText.tag_add("here", "1.0", "1.1")
    commandText.tag_configure("times", offset=70, font=('Arial Black', 40, "bold"))
    commandText.insert(INSERT,"Q","","*10","times")
    commandText.grid(row=0)
    commandText.place(x=80, y=200)
    
    img = ImageTk.PhotoImage(Image.open("media/sound.png"))
    panel = Label(window, image = img, borderwidth = 0)

    current_overlay_status = ""
    while True:
        time.sleep( 0.032 )
        filepath = COMMAND_FILE
        
        with open(filepath) as fp:  
            ctrl_shift_alt = fp.readline()
            sound = fp.readline()
            command = fp.readline()
            times = fp.readline()
            
            text.delete('1.0', END)
            text.insert(INSERT, sound)
            commandText.delete('1.0', END)
            
            if (times!="1"):
                commandText.insert(INSERT,command.rstrip("\n"),"","*" + times,"times")
            else:
                commandText.insert(INSERT,command.rstrip("\n"))                

            panel.place(x=0, y=20)    
            fp.close()

        window.update_idletasks()
        window.update()

loop_overlay()
#root = tk.Tk()

#l=Text(root)
#l.tag_configure("s", offset=5)
#l.insert(INSERT,"X","","2","s")
#l.grid(row=0)
#root.mainloop()
