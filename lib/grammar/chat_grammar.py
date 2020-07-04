from dragonfly import Grammar, CompoundRule, Integer, Choice, Repetition, Optional
from pyautogui import press, typewrite

quickCommands = {}
quickCommands["Good luck have fun"] = "gl hf"
quickCommands["This was a good game"] = "gg"
quickCommands["Good game"] = "gg"
quickCommands["Good game well played"] = "gg wp"

chatChoices = Choice( "quickcommand", quickCommands)

class ChatCommandRule(CompoundRule):
	spec = "<quickcommand>"
	extras = [chatChoices]
	callback = False
	
	def set_callback( self, callback ):
		self.callback = callback
	
	def _process_recognition(self, node, extras):
		typewrite( extras["quickcommand"], interval=0.1 )
		press( "enter" )
		
		if( self.callback ):
			self.callback()
