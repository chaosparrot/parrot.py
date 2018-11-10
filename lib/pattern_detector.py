import time
import pyautogui
from config.config import *
pyautogui.FAILSAFE = False

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
			
	def detect_strategy( self, action, config ):
		if( 'throttle' in config and self.throttle_detection( action, config['throttle'] ) ):
			return False

		strategy = config['strategy']
		label = config['sound']
		lastDict = self.predictionDicts[-1]
		
		detected = False
		if( strategy == 'single_tap' ):
			detected = ( self.above_percentage( lastDict, label, config['percentage'] ) and
				self.above_intensity( lastDict, config['intensity'] ) and
				self.rising_intensity( lastDict, self.predictionDicts[-2] ) )
				
		elif( strategy == 'rapid' ):
			detected = ( self.above_percentage( lastDict, label, config['percentage'] ) and
				self.above_intensity( lastDict, config['intensity'] ) )
				
		elif( strategy == 'continuous' ):
			if( self.throttle_detection( action, RECORD_SECONDS * 2 ) == False ):
				detected = ( self.above_percentage( lastDict, label, config['lowest_percentage'] ) and
					self.above_intensity( lastDict, config['lowest_intensity'] ) )
			else:
				detected = ( self.above_percentage( lastDict, label, config['percentage'] ) and
					self.above_intensity( lastDict, config['intensity'] ) )
			
		if( detected == True ):
			self.tickActions.append( action )
			self.timestamps[action] = self.currentTime
			print( "Detected " + action + " at " + str(self.currentTime) + "                                                           " )
			
		return detected
		
	# Detects if a label is the winning probability
	def is_winner( self, probabilityData, label ):
		return probabilityData[label]['winner']
		
	# Detects if a label has a probability above the given percentage
	def above_percentage( self, probabilityData, label, percentage ):
		return probabilityData[label]['percent'] >= percentage
		
	# Detects if a label has a probability above the given percentage
	def above_intensity( self, probabilityData, requiredIntensity ):
		return probabilityData['silence']['intensity'] >= requiredIntensity
		
	# Detect whether or not the sound has gotten louder
	def rising_intensity( self, probabilityDataB, probabilityDataA ):
		return probabilityDataB['silence']['intensity'] > probabilityDataA['silence']['intensity']

	# Detect whether or not the sound has gotten more silent
	def falling_intensity( self, probabilityDataB, probabilityDataA ):
		return probabilityDataB['silence']['intensity'] < probabilityDataA['silence']['intensity']
		
	# Detects if a label has a higher probability than the other
	def winner_over( self, probabilityData, labelA, labelB ):
		return probabilityData[labelA]['percent'] > probabilityData[label]['percent']
		
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
