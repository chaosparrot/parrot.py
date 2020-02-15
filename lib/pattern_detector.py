import time
import pyautogui
from config.config import *
import math
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

class PatternDetector:
    currentTime = time.time()
    predictionDicts = []
    timestamps = {}
    config = {}
    
    tickActions = []
    screenWidth = 0
    screenHeight = 0
    mouseX = 0
    mouseY = 0

    def __init__(self, config):
        self.config = config
        self.screenWidth, self.screenHeight = pyautogui.size()

    # Update the timestamp used for throttle detection
    # And set the prediction dicts to be used for detection
    def tick( self, predictionDicts, timestamp=None ):
        self.currentTime = timestamp if timestamp != None else time.time()
        self.mouseX, self.mouseY = pyautogui.position()
        self.predictionDicts = predictionDicts
        self.tickActions = []

    # Throttle the type of detection by a certain amount of milliseconds
    def throttle_detection( self, key, throttle_in_seconds ):
        if key not in self.timestamps:
            self.timestamps[key] = self.currentTime
            return False
        else:
            return ( self.currentTime - throttle_in_seconds ) < self.timestamps[ key ]
                
    # Detect an action using the strategy that was set in the configuration
    def detect( self, action ):
        if action not in self.config:
            return False
        else:
            return self.detect_strategy( action, self.config[action] )
        
    def detect_silence( self ):
        return self.predictionDicts[-1]['silence']['intensity'] < SILENCE_INTENSITY_THRESHOLD
        
    def is_throttled( self, action ):
        if action not in self.config:
            return False
        else:
            return self.throttle_detection( action, self.config[action]['throttle'] )
            
    def throttle( self, action ):
        self.timestamps[action] = self.currentTime
            
    def clear_throttle( self, action ):
        self.timestamps[action] = 0
        
    def set_throttle( self, action, throttle ):
        self.config[action]['throttle'] = throttle
        
    def add_tick_action( self, action ):
        self.tickActions.append( action )
        
    def deactivate_for( self, action, delay ):
        throttle = 0
        if( 'throttle' in self.config[action] ):
            throttle = self.config[action]['throttle']
        self.timestamps[action] = self.currentTime + delay - throttle

    def detect_strategy( self, action, config ):
        if( 'throttle' in config and self.throttle_detection( action, config['throttle'] ) ):
            return False

        strategy = config['strategy']
        label = config['sound']
        lastDict = self.predictionDicts[-1]
        
        detected = False
        if( strategy == 'single_tap' ):
            detected = ( self.above_percentage( lastDict, label, config['percentage'] ) and
                ( 'intensity' not in config or self.above_intensity( lastDict, config['intensity'] ) ) and 
                ( 'power' not in config or self.above_power( lastDict, config['power'] ) ) and
                self.rising_intensity( lastDict, self.predictionDicts[-2] ) )
                
        elif( strategy == 'rapid_intensity' ):
            detected = ( self.above_percentage( lastDict, label, config['percentage'] ) and
                self.above_intensity( lastDict, config['intensity'] ) )
                
        elif( strategy == 'rapid_power' ):
            detected = ( self.above_percentage( lastDict, label, config['percentage'] ) and
                self.above_power( lastDict, config['power'] ) )
                
        elif( strategy == 'frequency_threshold' ):
            detected = ( self.above_percentage( lastDict, label, config['percentage'] ) and
                self.above_power( lastDict, config['power'] ) )
                
            if( detected and 'above_frequency' in config ):
                detected = self.above_frequency( lastDict, config['above_frequency'] )
            
            if( detected and 'below_frequency' in config ):
                detected = self.below_frequency( lastDict, config['below_frequency'] )
        elif( strategy == 'continuous' ):
            if( self.throttle_detection( action, RECORD_SECONDS * 6 ) == True ):
                detected = ( self.above_percentage( lastDict, label, config['lowest_percentage'] ) and
                    self.above_intensity( lastDict, config['lowest_intensity'] ) )
            else:
                detected = ( self.above_percentage( lastDict, label, config['percentage'] ) and
                    self.above_intensity( lastDict, config['intensity'] ) )
                if( detected == True ):
                    self.timestamps[ action + "_start" ] = self.currentTime

        elif( strategy == 'continuous_power' ):
            if( self.throttle_detection( action, RECORD_SECONDS * 6 ) == True ):
                detected = ( self.above_percentage( lastDict, label, config['lowest_percentage'] ) and
                    self.above_power( lastDict, config['lowest_power'] ) )
            else:
                detected = ( self.above_percentage( lastDict, label, config['percentage'] ) and
                    self.above_power( lastDict, config['power'] ) )
                if( detected == True ):
                    self.timestamps[ action + "_start" ] = self.currentTime
                    
                    
        elif( strategy == 'combined_continuous' ):
            secondary_label = config['secondary_sound']        
            if( self.throttle_detection( action, RECORD_SECONDS * 6 ) == True ):
                detected = ( self.combined_above_percentage( lastDict, label, secondary_label, config['lowest_percentage'] ) and
                    self.above_intensity( lastDict, config['lowest_intensity'] ) )
            else:
                detected = ( self.combined_above_percentage( lastDict, label, secondary_label, config['percentage'] ) and
                    self.above_intensity( lastDict, config['intensity'] ) )
                if( detected == True ):
                    self.timestamps[ action + "_start" ] = self.currentTime

                    
        elif( strategy == 'combined' ):
            secondary_label = config['secondary_sound']
        
            detected = ( self.above_intensity( lastDict, config['intensity'] ) and
                self.combined_above_percentage( lastDict, label, secondary_label, config['percentage'] ) and
                self.above_ratio( lastDict, label, secondary_label, config['ratio'] ) )
        
        elif( strategy == 'combined_power' ):
            secondary_label = config['secondary_sound']
        
            detected = ( self.above_power( lastDict, config['power'] ) and
                self.combined_above_percentage( lastDict, label, secondary_label, config['percentage'] ) and
                self.above_ratio( lastDict, label, secondary_label, config['ratio'] ) )
                
        elif( strategy == 'combined_frequency' ):
            secondary_label = config['secondary_sound']
        
            detected = ( self.above_intensity( lastDict, config['intensity'] ) and
                self.below_frequency( lastDict, config['frequency'] ) and
                self.combined_above_percentage( lastDict, label, secondary_label, config['percentage'] ) and
                self.above_ratio( lastDict, label, secondary_label, config['ratio'] ) )
        elif( strategy == 'combined_quiet' ):
            secondary_label = config['secondary_sound']
        
            detected = ( self.below_intensity( lastDict, config['intensity'] ) and
                self.combined_above_percentage( lastDict, label, secondary_label, config['percentage'] ) and
                self.above_ratio( lastDict, label, secondary_label, config['ratio'] ) )
        
        
        if( detected == True ):
            self.tickActions.append( action )
            self.timestamps[action] = self.currentTime
            #print( "Detected " + action )
            
        return detected
        
    # Detects if a label is the winning probability
    def is_winner( self, probabilityData, label ):
        return probabilityData[label]['winner']
        
    # Detects if a label has a probability above the given percentage
    def above_percentage( self, probabilityData, label, percentage ):
        return probabilityData[label]['percent'] >= percentage
        
    # Detects if two labels have a combined probability above the given percentage
    def combined_above_percentage( self, probabilityData, label, secondary_label, percentage ):
        return ( probabilityData[label]['percent'] + probabilityData[secondary_label]['percent'] ) >= percentage
        
    # Detects if two labels have a combined probability above the given percentage
    def above_ratio( self, probabilityData, label, secondary_label, ratio ):
        return ( probabilityData[label]['percent'] / max( 1, probabilityData[secondary_label]['percent'] ) ) >= ratio        
        
    # Detects if a label has a probability above the given percentage
    def above_intensity( self, probabilityData, requiredIntensity ):
        return probabilityData['silence']['intensity'] >= requiredIntensity

    # Detects if a label has a probability above the given percentage
    def below_intensity( self, probabilityData, requiredIntensity ):
        return probabilityData['silence']['intensity'] < requiredIntensity
        
        
    # Detect whether or not the sound has gotten louder
    def rising_intensity( self, probabilityDataB, probabilityDataA ):
        return probabilityDataB['silence']['intensity'] > probabilityDataA['silence']['intensity']

    # Detect whether or not the sound has gotten more silent
    def falling_intensity( self, probabilityDataB, probabilityDataA ):
        return probabilityDataB['silence']['intensity'] < probabilityDataA['silence']['intensity']
        
    # Detects if a label has a higher probability than the other
    def winner_over( self, probabilityData, labelA, labelB ):
        return probabilityData[labelA]['percent'] > probabilityData[label]['percent']
        
    # Detects if a label is above a certain frequency
    def above_frequency( self, probabilityData, frequency ):
        return probabilityData['silence']['frequency'] >= frequency        
        
    # Detects if a label is above a certain power intensity
    def above_power( self, probabilityData, power ):
        return probabilityData['silence']['power'] >= power

    # Detects if a label is above a certain power intensity
    def below_power( self, probabilityData, power ):
        return probabilityData['silence']['power'] < power        
        
    # Detects if a label is below a certain frequency
    def below_frequency( self, probabilityData, frequency ):
        return probabilityData['silence']['frequency'] < frequency
        
    # Detects if the combined given labels are above this percentage
    def combined_percentage( self, probabilityData, labels, percentage ):
        combinedPercent = 0
        for label in labels:
            combinedPercent += probabilityData[label]['percent']
        
        return percent >= percentage
        
    # Detects if there are positive pitch changes between the last probabilities and the second to last probabilities
    # Within a certain threshold
    def pitch_up( self, probabilityDataB, probabilityDataA, threshold ):
        return ( probabilityDataB['silence']['frequency'] - threshold ) > probabilityDataA['silence']['frequency']
        
    # Detects if there are negative pitch changes between the last probabilities and the second to last probabilities
    # Within a certain threshold
    def pitch_down( self, probabilityDataB, probabilityDataA, threshold ):
        return ( probabilityDataB['silence']['frequency'] - threshold ) < probabilityDataA['silence']['frequency']
        
    # Detects if the pitch stays roughly the same within this threshold
    # Within a certain threshold
    def monotone( self, probabilityDataB, probabilityDataA, threshold ):
        return abs( probabilityDataB['silence']['frequency'] - probabilityDataA['silence']['frequency'] ) <= threshold
        
    # Detects in which quadrant the mouse is currently in
    # Counting goes as followed -
    # 1, 2, 3
    # 4, 5, 6
    # 7, 8, 9
    def detect_mouse_quadrant( self, widthSegments, heightSegments ):
        if( self.mouseX == 0 ):
            widthPosition = 0
        else:
            widthPosition = math.floor( self.mouseX / ( self.screenWidth / widthSegments ) )
        
        if( self.mouseY == 0 ):
            heightPosition = 0
        else:
            heightPosition = math.floor( self.mouseY /( self.screenHeight / heightSegments ) )

        quadrant = 1 + widthPosition + ( heightPosition * widthSegments )
        return quadrant

    # Detects on what edges the mouse is currently in
    def detect_mouse_screen_edge( self, threshold ):        
        edges = []
        if( self.mouseY <= threshold ):
            edges.append( "up" )
        elif( self.mouseY >= self.screenHeight - threshold ):
            edges.append( "down" )
        
        if( self.mouseX <= threshold ):
            edges.append( "left" )
        elif( self.mouseX >= self.screenWidth - threshold ):
            edges.append( "right" )
            
        return edges

    # Detects the X and Y position on the minimap
    # By taking the screen dimensions as the size for the minimap as if it were enlarged
    # For extra accuracy using eyetracking
    def detect_minimap_position( self, minimap_x, minimap_y, minimap_width, minimap_height ):
        ratioX = self.mouseX / self.screenWidth;
        ratioY = self.mouseY / self.screenHeight;
        
        minimapX = ( ratioX * minimap_width ) + minimap_x
        minimapY = ( ratioY * minimap_height ) + minimap_y
        return minimapX, minimapY
        
    ## Detect whether or not we are inside an area with our cursor
    def detect_inside_minimap( self, minimap_x, minimap_y, minimap_width, minimap_height ):
        return self.mouseX >= minimap_x and self.mouseY >= minimap_y and self.mouseX < minimap_x + minimap_width and self.mouseY < minimap_y + minimap_height
