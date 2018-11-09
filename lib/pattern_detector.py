import time


class PatternDetector:
	currentTime = time.time()
	predictionDicts = []
	timestamps = {}
	config = {}

	def __init__(self, config):
		self.config = config

	# Update the timestamp used for throttle detection
	# And set the prediction dicts to be used for detection
	def tick( self, predictionDicts ):
		self.currentTime = time.time()
		this.predictionDicts = predictionDicts

	# Throttle the type of detection by a certain amount of milliseconds
	def throttle_detection( self, key, throttle_in_milliseconds ):
		if key not in self.timestamps:
			self.timestamps[key] = currentTime
			return True
		else:
			return self.currentTime - throttle_in_milliseconds / 1000 ) > self.timestamps[ key ]:
				
	# Detect an action using the strategy that was set in the configuration
	def detect( action ):
		if action not in config:
			return False
		else:
			self.detect_strategy( action, config[action] )
		
	def detect_strategy( action, config ):
		# Detect if we should throttle the action	
		if( 'throttle_in_milliseconds' in config and self.throttle_detection( action, config['throttle_in_milliseconds'] ) == False:
			return False

		# DETECT!?
			
		if( 'throttle_in_milliseconds' in config ):
			self.timestamps[action] = currentTime