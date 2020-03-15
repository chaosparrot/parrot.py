from config.config import *
from lib.modes.base_mode import *

class TutorialMode(BaseMode):

    patterns = [
        {
            'name': 'loud',
            'sounds': ['noise', 'silence'],
            'threshold': {
                'percentage': 90,
                'below_frequency': 100,
                'power': 50000
            },
            'continual_threshold': {
                'percentage': 20,
                'power': 1000
            }
        }
    ]

    def handle_sounds( self, dataDicts ):
        if( self.detect('loud') ):
            self.press('a')