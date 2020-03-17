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
            if( self.detect_inside_area(0, 0, 400, 400 ):
                press('1')
            else:
			    press('2')