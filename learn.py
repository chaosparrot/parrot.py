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
import warnings
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import pandas
from scipy.fftpack import fft, rfft, fft2, dct
from python_speech_features import mfcc
from sklearn.manifold import TSNE
from ggplot import *

def hash_directory_to_number( setdir ):
	return float( int(hashlib.sha256( setdir.encode('utf-8')).hexdigest(), 16) % 10**8 )

# Get the full directories for the dataset
dir_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), "dataset")
data_directory_names = os.listdir( dir_path )
data_directories = list( map( lambda n: os.path.join( dir_path, n ), data_directory_names) )

# Add a label used for classifying the sounds
data_directories_label = list( map( hash_directory_to_number, data_directories ) )
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

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
	
def load_wav_files( directory, label, start, end ):
	category_dataset_x = []
	category_dataset_labels = []
	first_file = False
	for fileindex, file in enumerate(os.listdir(directory)):
		if ( file.endswith(".wav") and fileindex >= start and fileindex < end ):
			full_filename = os.path.join(directory, file)
			print( "Loading files for " + data_directory_names[ index ] + "... " + str(fileindex), end="\r" )
			
			# Load the WAV file and turn it into a onedimensional array of numbers
			fs, rawWav = scipy.io.wavfile.read( full_filename )
			chan1 = rawWav[:,0]
			chan2 = rawWav[:,1]
							
			# FFT is symmetrical - Only need one half of it to preserve memory
			#complexspectrum = fft( chan1 )
			#powerspectrum = np.abs( complexspectrum ) ** 2
			
			#f = np.linspace( samplerate, len( chan1 ), endpoint=False)
			#freqs = np.fft.fftfreq( len(chan1), 1 / 60 )
			#idx = np.argsort(freqs)
			
			#if( first_file ):
			#	plt.title( "Frequencies for file of " + data_directory_names[ index ] )
			#	plt.plot( f, abs( ft ) )
			#	plt.show()
			#	first_file = False
			
			mfcc_result1 = mfcc( chan1, samplerate=fs, nfft=1103 )
			mfcc_result2 = mfcc( chan2, samplerate=fs, nfft=1103 )
			data_row = []
			data_row.extend( mfcc_result1.ravel() )
			data_row.extend( mfcc_result2.ravel() )
			
			category_dataset_x.append( data_row )
			category_dataset_labels.append( label )

	print( "Loaded " + str( len( category_dataset_labels ) ) + " .wav files for category " + data_directory_names[ index ] )
	return category_dataset_x, category_dataset_labels
	

# Generate the training set and labels with them
dataset = []
dataset_x = []
dataset_labels = []

clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=123)
classifier = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=123)
#clf = MLPClassifier(hidden_layer_sizes=(527, 527, 527), solver='adam', activation='tanh', alpha=0.00001, learning_rate='invscaling', random_state=1, tol=0.0000001, verbose=True, max_iter=400)
#classifier = MLPClassifier(hidden_layer_sizes=(527, 527, 527), solver='adam', activation='tanh', alpha=0.00001, learning_rate='invscaling', random_state=1, tol=0.0000001, verbose=True, max_iter=400)
	
print( "Loading training set... " )
def partial_dataset_fitting( clf, start, end ):
	dataset_x = []
	dataset_labels = []
	for index, directory in enumerate( data_directories ):
		label = data_directories_label[ index ]
		cat_dataset_x, cat_dataset_labels = load_wav_files( directory, label, start, end )
		
		print( "Loading " + str( len( cat_dataset_labels ) ) + " .wav files for category " + data_directory_names[ index ] )
		#print( "Starting partial fitting..." )
		  
		dataset_x.extend( cat_dataset_x )
		dataset_labels.extend( cat_dataset_labels )
	
	print( "Loaded training set with " + str( len( dataset_x ) ) + " wav files")
	#if( start == 0 ):
		#clf.fit( dataset_x, dataset_labels ) 
		#for partial fit classes=np.unique(data_directories_label)

	#for i in range(50):
	#	clf.partial_fit( dataset_x, dataset_labels )

#partial_dataset_fitting( clf, 0, 400 )
#partial_dataset_fitting( clf, 50, 100 )
#partial_dataset_fitting( clf, 100, 150 )
#partial_dataset_fitting( clf, 150, 200 )
#partial_dataset_fitting( clf, 200, 250 )
#partial_dataset_fitting( clf, 250, 300 )

for index, directory in enumerate( data_directories ):
	label = data_directories_label[ index ]
	cat_dataset_x, cat_dataset_labels = load_wav_files( directory, label, 0, 1000 )
	dataset_x.extend( cat_dataset_x )
	dataset_labels.extend( cat_dataset_labels )


#classifier = RandomForestClassifier(n_estimators=25, max_depth=10, random_state=123)	
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

# Hyperdimension visualisation
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform( dataset_x, dataset_labels )


feat_cols = [ 'pixel'+str(i) for i in range(pandas.DataFrame(dataset_x).shape[1]) ]
df = pandas.DataFrame(dataset_x,columns=feat_cols)
df['label'] = dataset_labels
df['label'].apply(lambda i: str(i))

df_tsne = df
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]
chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=1) \
        + ggtitle("tSNE dimensions colored by digit")

		
print( chart )

#if( generate_confusion_matrix ):
#	plt.show()