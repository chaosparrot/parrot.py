from config.config import *
from lib.modes.base_mode import *

class VisualMode(BaseMode):

    latest_sound = ''
    
    def handle_input( self, dataDicts ):
        super().handle_input( dataDicts )
        self.update_command_file( dataDicts )

        return self.detector.tickActions
                
    def press( self, key, print_key = False ):
        super().press( key )
        self.print_key( key, print_key )
        
    def hold( self, key, print_key = False ):
        super().hold( key )
        self.print_key( key, print_key )
        
    def leftclick( self ):
        super().leftclick()
        self.print_key( 'LMB', 'LMB' )

    def rightclick( self ):
        super().rightclick()
        self.print_key( 'RMB', 'RMB' )
        
    def drag_mouse( self ):
        super().drag_mouse()
        self.print_key( 'drag', 'Mouse drag' )        
        
    def print_key( self, key, print_key = False ):
        if (print_key == False):
            print_key = key.upper()        
        self.detector.add_tick_action( print_key )
        
    def detect( self, key ):
        detected = super().detect(key)
        
        if (detected):
            sound = self.detector.patterns[key]['sounds'][0]
            self.latest_sound = "/" + sound.replace("general_", "").replace("vowel_", "")\
                .replace("sibilant_", "").replace("thrill_", "").replace("sound_", "").replace("fricative_", "").replace("_alveolar", "").replace("approximant_", "") + "/"
        
        return detected
        
    def update_command_file( self, dataDicts ):
        with open(COMMAND_FILE, 'r+') as fp:
        
            # Read initial data first
            ctrl_shift_alt = fp.readline()
            sound = fp.readline().rstrip("\n")
            command = fp.readline().rstrip("\n")
            times = fp.readline().rstrip("\n")
            if (times == ""):
                times = 0                

            ctrl_shift_alt = ""
            if (self.inputManager.toggle_keys['ctrl']):
                ctrl_shift_alt = "ctrl"
            if (self.inputManager.toggle_keys['shift']):
                ctrl_shift_alt = ctrl_shift_alt + "shift"
            if (self.inputManager.toggle_keys['alt']):
                ctrl_shift_alt = ctrl_shift_alt + "alt"                    
            
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
            fp.write(ctrl_shift_alt + '\n')
            fp.write(sound + '\n')
            fp.write(command + '\n')
            fp.write(str(times))
            fp.close()