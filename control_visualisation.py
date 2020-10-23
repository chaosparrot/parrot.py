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
    root.wm_title("Tk")

    window = tk.Toplevel(root)
    window.wm_title("Visualisation")    
    window.configure(bg="#000000")
    ##window.overrideredirect(1) #Remove border
    ##window.attributes('-topmost', -1) # Keep on top
    window.geometry('%dx%d+%d+%d' % (500, 700, 0, 0))
    ##window.attributes("-transparentcolor", "#FFFFFF") # Remove background
    timestamp = time.time()
    
    text = tk.Text(window, height=1, width=20, relief="flat")
    text.configure(bg="#000000", fg="#FFFFFF", padx=20, pady=20, font=("Arial Black", 40, "bold"))
    text.insert(INSERT, "vowel_ah")
    text.place(x=150, y=0)
    
    commandText = tk.Text(window, relief="flat", width=20, height=1)
    commandText.configure(bg="#000000", fg="#FFFFFF", font=("Arial Black", 120, "bold"))
    commandText.tag_add("here", "1.0", "1.1")
    commandText.tag_configure("times", offset=70, font=('Arial Black', 40, "bold"))
    commandText.insert(INSERT,"Q","","*10","times")
    commandText.grid(row=0)
    commandText.place(x=80, y=80)
    
    holdText = tk.Text(window, relief="flat", width=20, height=1)
    holdText.configure(bg="#000000", fg="#FFFFFF", font=("Arial Black", 35, "bold"))
    holdText.insert(INSERT," CTRL ","CTRL"," SHIFT ","SHIFT", " ALT ", "ALT")
    holdText.tag_configure("CTRL", background="#FF0000", foreground="#000000")
    holdText.tag_configure("SHIFT", background="#0000FF", foreground="#000000")
    holdText.tag_configure("ALT", background="#FFF000", foreground="#000000")
    holdText.place(x=0, y=430)
    
    # Draw the controller nipple    
    canvas_size = 85
    canvas = Canvas(window, width=canvas_size, height=canvas_size, bg='#000000', highlightthickness=0)
    canvas.place(x=380, y=420 - canvas_size)
    canvas.create_oval(3, 3, canvas_size - 3, canvas_size - 3, outline="#333", fill="#444", width=2)
    
    width = 40
    height = 40
    x = canvas_size / 2 - width / 2
    y = canvas_size / 2 - height / 2
    points = [[x, y + 0.5 * height], [x + 0.15 * width, y + 0.15 * height], [x + 0.5 *width, y], [x + 0.85 * width, y + 0.15 * height],
        [x + width, y + 0.5 * height], [x + 0.85 * width, y + 0.85 * height], [x + 0.5 * width, y + height], [x + 0.15 * width, y + 0.85 * height], [x, y + 0.5 * height]]
    
    canvas.create_polygon(points, fill="#111") 
    basePosX = canvas_size / 2
    basePosY = canvas_size / 2    
    canvasOffsetX = basePosX
    canvasOffsetY = basePosY
    nipple_outer = canvas.create_oval(canvasOffsetX - 25, canvasOffsetY - 25, canvasOffsetX + 25, canvasOffsetY + 25, outline="#555", fill="#999", width=2)
    nipple_middle = canvas.create_oval(canvasOffsetX - 18, canvasOffsetY - 18, canvasOffsetX + 18, canvasOffsetY + 18, outline="#555", fill="#999", width=2)
    nipple_inner = canvas.create_oval(canvasOffsetX - 13, canvasOffsetY - 13, canvasOffsetX + 13, canvasOffsetY + 13, outline="#555", fill="#999", width=2)
    nipple_dot = canvas.create_oval(canvasOffsetX - 1, canvasOffsetY - 1, canvasOffsetX + 1, canvasOffsetY + 1, outline="#000", fill="#000", width=1)
    
    img = ImageTk.PhotoImage(Image.open("media/sound.png"))
    panel = Label(window, image = img, borderwidth = 0)

    current_overlay_status = ""
    while True:
        time.sleep( 0.032 )
        filepath = COMMAND_FILE
                
        with open(filepath) as fp:  
            sound = fp.readline()
            times = fp.readline()
            command = fp.readline()
            held_keys = fp.readline()            
            
            text.delete('1.0', END)
            text.insert(INSERT, sound)
            commandText.delete('1.0', END)
            
            canvas.delete( nipple_outer )
            canvas.delete( nipple_middle )
            canvas.delete( nipple_inner )
            canvas.delete( nipple_dot )
            
            canvasOffsetX = basePosX
            canvasOffsetY = basePosY
            if ( 'left' in held_keys ):
                canvasOffsetX = canvasOffsetX - canvas_size / 5
            elif ( 'right' in held_keys ):
                canvasOffsetX = canvasOffsetX + canvas_size / 5
                
            if ( 'up' in held_keys ):
                canvasOffsetY = canvasOffsetY - canvas_size / 5
            elif ( 'down' in held_keys ):
                canvasOffsetY = canvasOffsetY + canvas_size / 5                
            
            # Draw the controller nipple
            nipple_outer = canvas.create_oval(canvasOffsetX - 25, canvasOffsetY - 25, canvasOffsetX + 25, canvasOffsetY + 25, outline="#555", fill="#999", width=2)
            nipple_middle = canvas.create_oval(canvasOffsetX - 18, canvasOffsetY - 18, canvasOffsetX + 18, canvasOffsetY + 18, outline="#555", fill="#999", width=2)
            nipple_inner = canvas.create_oval(canvasOffsetX - 13, canvasOffsetY - 13, canvasOffsetX + 13, canvasOffsetY + 13, outline="#555", fill="#999", width=2)
            nipple_dot = canvas.create_oval(canvasOffsetX - 2, canvasOffsetY - 2, canvasOffsetX + 2, canvasOffsetY + 2, outline="#000", fill="#000", width=2)

            if( len(command) > 3 ):
                commandText.configure(font=("Arial Black", 50, "bold"))
                commandText.tag_configure("times", offset=10, font=('Arial Black', 40, "bold"))
                commandText.place(x=120, y=230)
                if( len(command) > 8 ):
                    commandText.place(x=40, y=230)
            else:
                commandText.configure(font=("Arial Black", 120, "bold"))
                commandText.tag_configure("times", offset=70, font=('Arial Black', 40, "bold"))
                commandText.place(x=160, y=150)
                
            if ("ctrl" in held_keys):
                holdText.tag_configure("CTRL", background="#FF0000", foreground="#000000")
            else:
                holdText.tag_configure("CTRL", background="#000000", foreground="#000000")            

            if ("shift" in held_keys):
                holdText.tag_configure("SHIFT", background="#0000FF", foreground="#000000")
            else:
                holdText.tag_configure("SHIFT", background="#000000", foreground="#000000")

            if ("alt" in held_keys):
                holdText.tag_configure("ALT", background="#FFF000", foreground="#000000")
            else:
                holdText.tag_configure("ALT", background="#000000", foreground="#000000")
            
            if (times!="1"):
                commandText.insert(INSERT,command.rstrip("\n"),"","*" + times,"times")
            else:
                commandText.insert(INSERT,command.rstrip("\n"))                

            panel.place(x=80, y=20)    
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
