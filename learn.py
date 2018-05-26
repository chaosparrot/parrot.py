import scipy
import scipy.io.wavfile
import os
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.externals import joblib
import time

import itertools
import matplotlib.pyplot as plt
import numpy as np
import hashlib


from scipy.fftpack import fft, rfft, fft2


def hash_directory_to_number( setdir ):
	return float( int(hashlib.sha256( setdir.encode('utf-8')).hexdigest(), 16) % 10**8 )

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
	
def load_wav_files( directory, label ):
	category_dataset_x = []
	category_dataset_labels = []
	for file in os.listdir(directory):
		if file.endswith(".wav"):
			full_filename = os.path.join(directory, file)
			
			# Load the WAV file and turn it into a onedimensional array of numbers
			rawWav = scipy.io.wavfile.read( full_filename )[ 1 ]
			chan1 = rawWav[:,0]
			
			datarow = np.abs( fft( chan1 ) ) ** 2
						
			category_dataset_x.append( datarow )
			category_dataset_labels.append( label )

	return category_dataset_x, category_dataset_labels
	

# Get the full directories for the dataset
dir_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), "dataset")
data_directory_names = os.listdir( dir_path )
data_directories = list( map( lambda n: os.path.join( dir_path, n ), data_directory_names) )

# Add a label used for classifying the sounds
data_directories_label = list( map( hash_directory_to_number, data_directories ) )

# Generate the training set and labels with them
dataset = []
dataset_x = []
dataset_labels = []

print( "Loading training set... " )
clf = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=123)
for index, directory in enumerate( data_directories ):
	label = data_directories_label[ index ]
	cat_dataset_x, cat_dataset_labels = load_wav_files( directory, label )
	
	print( "Loading " + str( len( cat_dataset_labels ) ) + " .wav files for category " + data_directory_names[ index ] )
	dataset_x.extend( cat_dataset_x )
	dataset_labels.extend( cat_dataset_labels )
	
print( "Loaded training set with " + str( len( dataset_x ) ) + " wav files")

while( True ):
	print( "Weeeee" )

classifier = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=123)

#classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#classifier = svm.SVC(C=1.0, gamma=0.01)
#classifier = DecisionTreeClassifier(random_state=0)

check_accuracy = False
if( check_accuracy ):
	print( "Beginning cross validation for accuracy prediction" )
	scores = cross_val_score(classifier, dataset_x, dataset_labels, cv=10)
	print( "Cross validation done!" )
	print( scores )
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Start classifying and checking for confusion
generate_confusion_matrix = False
if( generate_confusion_matrix ):
	X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_labels, random_state=1)
	classifier.fit( X_train, y_train )
	start_time = time.time() * 1000
	y_pred = classifier.predict( X_test )
	end_time = time.time() * 1000
	print( "Predicted " + str( len( X_test ) ) + " samples in " + str( int( end_time - start_time ) ) + " ms" )

	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
	print( cnf_matrix )
	
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=data_directory_names,
						  title='Confusion matrix')

# Persist the complete classifier					  
clf.fit( dataset_x, dataset_labels )
joblib.dump( clf, "train.pkl" )

if( generate_confusion_matrix ):
	plt.show()