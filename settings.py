from config.config import *
from lib.record_data import record_sound
from lib.learn_data import learn_data
from lib.test_data import test_data
from lib.setup_config import setup_config
from lib.combine_models import combine_models

def root_navigation( first):
	if( first ):
		print( "Welcome to Parrot.PY setup!" )
		print( "----------------------------" )
	print( "Enter one of the buttons below and press enter to start" )
	print( " - [R] for recording" )
	print( " - [L] for learning the data" )
	print( " - [C] for combining multiple models into one model" )
	print( " - [A] for analyzing the performance of the models" )
	print( " - [S] for changing settings" )
	print( " - [X] for exiting setup" )
	
	select_mode()

def select_mode():
	setup_mode = input("")
	if( setup_mode.lower() == 'r' ):
		record_sound()
		root_navigation( False )
	elif( setup_mode.lower() == 'l' ):
		learn_data()
		root_navigation( False )
	elif( setup_mode.lower() == 'c' ):
		combine_models()
		root_navigation( False )
	elif( setup_mode.lower() == 'a' ):
		test_data( True )
		root_navigation( False )
	elif( setup_mode.lower() == 's' ):
		setup_config( True )
		root_navigation( False )
	elif( setup_mode.lower() == 'x' ):
		print( "Goodbye." )
	else:
		select_mode()
	
root_navigation( True )