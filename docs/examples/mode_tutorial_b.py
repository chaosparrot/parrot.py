from config.config import *
from lib.modes.base_mode import *

class TutorialMode(BaseMode):
    use_direct_keys = False
    input_release_lag = 0.0

    def handle_sounds( self, dataDicts ):
        if( not self.detect_silence() ):
            #####
            
            #self.press('a')
            
            #self.hold('a')
            
            #self.leftclick()
            #self.rightclick()
            
            #self.drag_mouse()
            
            #self.hold('ctrl')
            #self.press('a')
            #self.release('ctrl')
            
            #####
            return