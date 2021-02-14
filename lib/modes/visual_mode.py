from lib.modes.base_mode import *
import lib.ipc_manager as ipc_manager

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
        for key in self.inputManager.toggle_keys:
            ipc_manager.setButtonState(key, self.inputManager.toggle_keys[key])
        ipc_manager.setSoundName(self.latest_sound)
        if (len(self.detector.tickActions) > 0 ):
            new_command = self.detector.tickActions[-1]
            ipc_manager.setActionName(new_command)