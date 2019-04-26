from config.config import *
from sklearn.externals import joblib
from lib.hierarchial_classifier import *

def combine_models():
	classifier_map = {
		'main':  joblib.load( CLASSIFIER_FOLDER + "/train_narud_main.pkl" ),
		'category_vowels': joblib.load( CLASSIFIER_FOLDER + "/train_narud_vowels.pkl" ),
		'category_noise': joblib.load( CLASSIFIER_FOLDER + "/train_narud_noise.pkl" )
	}

	classifier = HierarchialClassifier( classifier_map )
	print( "Classifier contains the following prediction classes: ", classifier.classes_ )
	
	classifier_name = "train_narud_complete"
	classifier_filename = CLASSIFIER_FOLDER + "/" + classifier_name + ".pkl"
	joblib.dump( classifier, classifier_filename )