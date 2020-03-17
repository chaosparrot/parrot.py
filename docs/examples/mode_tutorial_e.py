from config.config import *
from lib.modes.base_mode import *

class TutorialMode(BaseMode):

    patterns = [
        {
            'name': 'loud',
            'sounds': ['noise'],
            'threshold': {
                'percentage': 90,
            },
            'throttle': {
                'loud': 0.3
            }
        }
    ]

    def handle_sounds( self, dataDicts ):
        if( self.detect('loud') ):
            if( self.quadrant3x3 == TOPLEFT ):
                press('1')
            elif( self.quadrant3x3 == TOPMIDDLE ):
                press('2')
            elif( self.quadrant3x3 == TOPRIGHT ):
                press('3')
            elif( self.quadrant3x3 == CENTERLEFT ):
                press('4')
            elif( self.quadrant3x3 == CENTERMIDDLE ):
                press('5')
            elif( self.quadrant3x3 == CENTERRIGHT ):
                press('6')
            elif( self.quadrant3x3 == BOTTOMLEFT ):
                press('7')
            elif( self.quadrant3x3 == BOTTOMMIDDLE ):
                press('8')
            elif( self.quadrant3x3 == BOTTOMRIGHT ):
                press('9')