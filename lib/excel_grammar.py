from dragonfly import Grammar, CompoundRule, Integer, Choice, Repetition, Optional
from pyautogui import press, hotkey, click, scroll, typewrite, moveRel, moveTo, position, keyUp, keyDown, mouseUp, mouseDown
import pyperclip

natoAlphabet = Choice("alphabet", {
				"Alpha": 1,
				"Bravo": 2,
				"Charlie": 3,
				"Delta": 4,
				"Echo": 5,
				"Foxtrot": 6,
				"George": 7,
				"Hotel": 8,
				"India": 9,
				"Juliet": 10,
				"Kilo": 11,
				"Lima": 12,
				"Mike": 13,
				"November": 14,
				"Oscar": 15,
				"Papa": 16,
				"Quebec": 17,
				"Romeo": 18,
				"Sierra": 19,
				"Tango": 20,
				"Uniform": 21,
				"Victor": 22,
				"Whiskey": 23,
				"X-ray": 24,
				"Yankee": 25,
				"Zulu": 26,
				}
			)
			
def natoNumberToLetter( number ):
	letters = ['','a','b','c','d','e','f','g','h','i','j','k','l','m','n',
		'o','p','q','r','s','t','u','v','w','x','y','z']
	return letters[ number ]

class CopyRowRule(CompoundRule):
	spec = "Do you copy over"

	def _process_recognition(self, node, extras):
		hotkey('ctrl', 'c')

class ShiftRule(CompoundRule):
	spec = "<shifttype>"
	extras = [Choice("shifttype", {'Engage': 'Engage', 'Disengage': 'Disengage'})]	
	
	def _process_recognition(self, node, extras):
	
		if( extras['shifttype'] == "Engage" ):
			keyDown('shift')
		else:
			keyUp("shift")		
		
class PasteRule(CompoundRule):
	spec = "Roger that"

	def _process_recognition(self, node, extras):
		hotkey('ctrl', 'v', interval = 0.15)

class MoveRule(CompoundRule):
	spec = "<type> <direction> <n>"
	extras = [Choice("type", {'Break': 'Break', 'Move': 'Move'}), Optional(Choice("direction", {'left': 'left', 'up': 'up', 'down': 'down', 'right': 'right'}), "direction"), Optional( Repetition(Integer("n", 0, 10), 0, 10, "n"), "n")]

	def _process_recognition(self, node, extras):
		numbers = extras["n"]
		direction = extras["direction"]
		if( direction == None ):
			direction = 'down'
		
		total = "".join( str(x) for x in numbers )
		if( total == "" ):
			total = "1"
		total = int( total )

		presses = []
		if( type == "Break" ):
			presses.append( 'home' )

		for i in range(total):
			presses.append( direction )
			
		press(presses)
		
class CorrectionRule(CompoundRule):
	spec = "correction"

	def _process_recognition(self, node, extras):
		hotkey('ctrl', 'z')
		
class ColumnNumberPrintRule(CompoundRule):
	spec = "<alphabet> <n> stop"
	extras = [Repetition(Integer("n", 0, 10), 0, 10, "n"), natoAlphabet]

	def _process_recognition(self, node, extras):
		natoNumber = extras["alphabet"]
		
		numbers = extras["n"]
		
		# Move to the correct column
		press('home')
		rightPresses = []
		for i in range(natoNumber - 1):
			rightPresses.append('right')
		
		press( rightPresses )
		typewrite( "".join( str(x) for x in numbers ) )
		
		# Make sure we keep our row selected
		press(["right", "left"])
		
class ColumnModePrintRule(CompoundRule):
	spec = "<alphabet> <n> stop"
	extras = [Repetition(Integer("n", 0, 10), 0, 10, "n"), Repetition(natoAlphabet, 2, 3, "alphabet" )]

	def _process_recognition(self, node, extras):
		natoNumbers = extras["alphabet"]
		numbers = extras["n"]
		
		# Move to the correct column
		press('home')
		rightPresses = []
		for i in range(natoNumbers[0] - 1):
			rightPresses.append('right')
		press( rightPresses )
		
		if( natoNumberToLetter( natoNumbers[1] ) == "n" ):
			typewrite( "".join( str(x) for x in numbers ) )
		elif( natoNumberToLetter( natoNumbers[1] ) == "t" ):
			if( len( numbers ) == 3 ):
				typewrite( ".0" + str(numbers[0]) + ":" + str(numbers[1]) + str(numbers[2]) + ":00" ) 
			elif( len( numbers ) == 4 ):
				typewrite( "." + str(numbers[0]) + str(numbers[1]) + ":" + str(numbers[2]) + str(numbers[3]) + ":00" )
		elif( natoNumberToLetter( natoNumbers[1] ) == "d" ):
			if( len( numbers ) == 3 ):
				typewrite( "0" + str(numbers[0]) + " - " + str(numbers[1]) + str(numbers[2]) )
			elif( len( numbers ) == 4 ):
				typewrite( str(numbers[0]) + str(numbers[1]) + " - " + str(numbers[2]) + str(numbers[3]) )

		
		# Make sure we keep our row selected
		press(["right", "left"])