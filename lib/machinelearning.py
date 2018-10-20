import hashlib
import scipy
import scipy.io.wavfile
from scipy.fftpack import fft, rfft, fft2, dct
from python_speech_features import mfcc
import time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

def feature_engineering( wavFile ):
	fs, rawWav = scipy.io.wavfile.read( wavFile )
	chan1 = rawWav[:,0]
				
	mfcc_result1 = mfcc( chan1, samplerate=fs, nfft=1103 )
	data_row = []
	data_row.extend( mfcc_result1.ravel() )
	return data_row
	
def get_label_for_directory( setdir ):
	return float( int(hashlib.sha256( setdir.encode('utf-8')).hexdigest(), 16) % 10**8 )

def cross_validation( classifier, dataset, labels):
	return cross_val_score(classifier, dataset, labels, cv=5)

def average_prediction_speed( classifier, dataset_x ):
	start_time = time.time() * 1000
	classifier.predict( dataset_x )
	end_time = time.time() * 1000
	return int( end_time - start_time ) / len(dataset_x)
	
def create_confusion_matrix(classifier, dataset_x, dataset_labels, all_labels):
	X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_labels, random_state=1)
	classifier.fit( X_train, y_train )
	y_pred = classifier.predict( X_test )

	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
	plot_confusion_matrix(cnf_matrix, all_labels )
	
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
    plt.ylabel('True category')
    plt.xlabel('Predicted category')
    plt.show()
	
#def generate_tnse( dataset_x, dataset_labels ):
	#tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	#tsne_results = tsne.fit_transform( dataset_x, dataset_labels )

	#feat_cols = [ 'pixel'+str(i) for i in range(pandas.DataFrame(dataset_x).shape[1]) ]
	#df = pandas.DataFrame(dataset_x,columns=feat_cols)
	#df['label'] = dataset_labels
	#df['label'].apply(lambda i: str(i))

	#df_tsne = df
	#df_tsne['x-tsne'] = tsne_results[:,0]
	#df_tsne['y-tsne'] = tsne_results[:,1]
	#chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
	#		+ geom_point(size=70,alpha=1) \
	#		+ ggtitle("tSNE dimensions colored by digit")
		
	#print( chart )