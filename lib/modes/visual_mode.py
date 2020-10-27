from config.config import *
from lib.modes.base_mode import *
import re

class VisualMode(BaseMode):

    latest_sound = ""
    
    def handle_input( self, dataDicts ):
        super().handle_input( dataDicts )
        self.update_command_file( dataDicts )

        return self.detector.tickActions
                
    def press( self, key, print_key = False ):
        super().press( key )
        self.print_key( key, print_key )
                        
    def leftclick( self ):
        super().leftclick()
        self.print_key( "LMB", "LMB" )

    def rightclick( self ):
        super().rightclick()
        self.print_key( "RMB", "RMB" )
        
    def drag_mouse( self ):
        super().drag_mouse()
        self.print_key( "drag", "Mouse drag" )        
        
    def print_key( self, key, print_key = False ):
        if (print_key == False):
            print_key = key.upper()        
        self.detector.add_tick_action( print_key )
        
    def detect( self, key ):
        detected = super().detect(key)
        
        if (detected and key not in self.toggles):
            sound = self.detector.patterns[key]["sounds"][0]
            self.latest_sound = "/" + sound.replace("general_", "").replace("vowel_", "").replace("stop_", "").replace("nasal_", "")\
                .replace("sibilant_", "").replace("thrill_", "").replace("sound_", "").replace("fricative_", "").replace("_alveolar", "").replace("approximant_", "") + "/"
        
        return detected
        
    def update_command_file( self, dataDicts ):
        with open(COMMAND_FILE, "r+") as fp:
        
            # Read initial data first
            sound = fp.readline().rstrip("\n")
            times = fp.readline().rstrip("\n")
            times = re.sub('[^0-9]','', times)
            command = fp.readline().rstrip("\n")
            held_keys = fp.readline().rstrip("\n")
            if (times == ""):
                times = 0

            held_keys = ""
            if (self.inputManager.toggle_keys["ctrl"]):
                held_keys = "ctrl"
            if (self.inputManager.toggle_keys["shift"]): 
                held_keys = held_keys + "shift"
            if (self.inputManager.toggle_keys["alt"]):
                held_keys = held_keys + "alt"
            if (self.inputManager.toggle_keys["left"]):
                held_keys = held_keys + "left"
            if (self.inputManager.toggle_keys["up"]):
                held_keys = held_keys + "up"
            if (self.inputManager.toggle_keys["right"]):
                held_keys = held_keys + "right"
            if (self.inputManager.toggle_keys["down"]):
                held_keys = held_keys + "down"
            
            if (len(self.detector.tickActions) > 1):
                sound = self.latest_sound
                
                new_command = self.detector.tickActions[-1]
                if (new_command == command):
                    times = int(times) + 1
                else:
                    times = 1
                command = new_command
                fp.truncate(0)

            # Start writing new information
            fp.seek(0)
            fp.write(sound + "\n")
            fp.write(str(times) + "\n")
            fp.write(command + "\n")
            fp.write(held_keys)
            fp.close()