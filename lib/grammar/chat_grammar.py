from dragonfly import Grammar, CompoundRule, Integer, Choice, Repetition, Optional
from pyautogui import press, typewrite

quickCommands = {}
quickCommands["Good luck have fun"] = "gl hf"
quickCommands["Good game"] = "gg"
quickCommands["Good game well played"] = "gg wp"

chatChoices = Choice( "quickcommand", quickCommands)
		
class ChatCommandRule(CompoundRule):
	spec = "chat <quickcommand>"
	extras = [chatChoices]
	callback = False
	
	def _process_recognition(self, node, extras):
		press( "enter" )
		command = quickCommands[extras["quickcommand"]]
		typewrite( command )
		press( "enter" )
		