from config.config import *
from lib.modes.base_mode import *

class TutorialMode(BaseMode):

    patterns = [
        {
            'name': 'loud',
            'sounds': ['noise'],
            'threshold': {
                'percentage': 90,
                'power': 50000
            },
            'throttle': {
                'loud': 0.3
            }
        },
        {
            'name': 'louder',
            'sounds': ['noise'],
            'threshold': {
                'percentage': 90,
                'power': 100000
            },
            'throttle': {
                'louder': 1.0,
                'loud': 1.0
            }
        }
    ]

    def handle_sounds( self, dataDicts ):
        if( self.detect('loud') ):
            self.press('a')
        elif( self.detect('louder') ):
            self.press('b')
