from config.config import *
import os
from lib.modes.base_mode import *

class TutorialMode(BaseMode):

    patterns = {
        'test': {
            'strategy': 'rapid_power',
            'sound': 'sibilant_s',
            'power': 30000,
            'percentage': 80
        }
    }

    def handle_sounds( self, dataDicts ):
        if( self.detect('test') ):
            self.drag_mouse()
        return