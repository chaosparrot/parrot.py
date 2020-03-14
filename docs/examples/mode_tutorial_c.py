from config.config import *
from lib.modes.base_mode import *

class TutorialMode(BaseMode):

    patterns = {
        'loud': {
            sounds: ['noise'],
            threshold: {
                'percentage': 90,
            },
            throttle: {
                'loud': 0.3,
            }
        },
        'louder': {
            sounds: ['noise'],
            threshold: {
                'percentage': 90,
                'power': 50000
            },
            throttle: {
                'louder': 0.3,
                'loud': 0.3
            }
        }
    }

    def handle_sounds( self, dataDicts ):
        if( self.detect('louder') ):
            self.press('b')
        elif( self.detect('loud') ):
            self.press('a')
            