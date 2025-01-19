from config.config import *
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
from lib.machinelearning import *
from sklearn.neural_network import *
from lib.combine_models import define_settings, get_current_default_settings
from lib.audio_model import AudioModel
from lib.load_data import load_sklearn_data, load_pytorch_data, load_sequential_pytorch_data

def learn_data():
    dir_path = os.path.join( os.path.dirname( os.path.dirname( os.path.realpath(__file__)) ), DATASET_FOLDER)    
    data_directory_names = [directory for directory in os.listdir( dir_path ) if os.path.isdir(dir_path + "/" + directory)]
    
    print( "-------------------------" )
    if (len(data_directory_names) < 2):
        print("You haven't recorded enough different sounds yet in " + DATASET_FOLDER + "! Make sure to record at least two different sounds before you train a model")
        print( "-------------------------" )
        return
    else:
        print( "Time to learn the audio files")
        print( "This script is going to analyze all your generated audio snippets" )
        print( "And attempt to learn the sounds to their folder names" )
        print( "-------------------------" )
    
    settings = define_settings(get_current_default_settings())
    
    clf_filename = input("Insert the model name ( empty is '" + DEFAULT_CLF_FILE + "' ) ")
    if( clf_filename == "" ):
        clf_filename = DEFAULT_CLF_FILE
    clf_filename += '.pkl'
    
    print( "Type the algorithm that you wish to use for recognition")
    print( "- [R] Random Forest ( SKLEARN - For quick verification )" )
    if( PYTORCH_AVAILABLE ):
        print( "- [A] Audio Net ( Neural net in Pytorch - Required by TalonVoice )" )
        print( "- [S] Sequential Audio Net ( EXPERIMENTAL Gru net in Pytorch )" )
    print( "- [M] Multi Layer Perceptron ( Neural net in SKLEARN )" )
    print( "- [X] Exit the learning" )

    
    model_type = input("")
    while model_type == "":
        model_type = input("")

    if( model_type.lower() == "r" ):
        classifier = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=123)
        print( "Selected random forest!")        
        
        fit_sklearn_classifier( classifier, dir_path, clf_filename, settings )        
    elif( model_type.lower() == "m" ):
        classifier = MLPClassifier(activation='relu', early_stopping=True,
              epsilon=1e-08, hidden_layer_sizes=(1024, 1024, 1024, 512),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=1000, momentum=0.9,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=True, warm_start=False)
        print( "Selected multi layer perceptron!")
        
        fit_sklearn_classifier( classifier, dir_path, clf_filename, settings )
    elif( model_type.lower() == "x" ):
        return
    elif( model_type.lower() == "a" and PYTORCH_AVAILABLE ):
        print( "Selected Audio Net!")
        print( "How many nets do you wish to train at the same time? ( Default is 3 )" )
        net_count = input("")
        if ( net_count == "" ):
            net_count = 3
        else:
            net_count = int(net_count)

        # Import pytorch related thins here to make sure pytorch isn't a hard requirement
        from lib.audio_net import AudioNetTrainer
        from lib.audio_dataset import AudioDataset

        print( "--------------------------" )
        dataset_labels = determine_labels( dir_path )
        print( "--------------------------" )
        data = load_pytorch_data(dataset_labels, settings["FEATURE_ENGINEERING_TYPE"])
        dataset = AudioDataset( data )
        trainer = AudioNetTrainer(dataset, net_count, settings)

        print( "Learning the data..." )
        trainer.train( clf_filename )
    elif( model_type.lower() == "s" and PYTORCH_AVAILABLE ):
        print( "Selected Sequential Audio Net!")
        print( "How many nets do you wish to train at the same time? ( Default is 3 )" )
        net_count = input("")
        if ( net_count == "" ):
            net_count = 3
        else:
            net_count = int(net_count)

        # Import pytorch related thins here to make sure pytorch isn't a hard requirement
        from lib.sequential_audio_net import SequentialAudioNetTrainer
        from lib.sequential_audio_dataset import SequentialAudioDataset

        print( "--------------------------" )
        dataset_labels = determine_labels( dir_path )
        print( "--------------------------" )
        data = load_sequential_pytorch_data(dataset_labels, settings["FEATURE_ENGINEERING_TYPE"])
        dataset = SequentialAudioDataset( data )
        trainer = SequentialAudioNetTrainer(dataset, net_count, settings)

        print( "Learning the data..." )
        trainer.train( clf_filename )

def fit_sklearn_classifier( classifier,  dir_path, clf_filename, settings ):    
    print( "--------------------------" )
    filtered_data_directory_names = determine_labels( dir_path )
    dataX, dataY, directory_names = load_sklearn_data( filtered_data_directory_names, settings['FEATURE_ENGINEERING_TYPE'] )
    print( "--------------------------" )

    print( "Learning the data..." )
    classifier.fit( dataX, dataY )
    print( "Data analyzed!               " )

    persisted_classifier = AudioModel( settings, classifier )
    
    print( "Saving the model to " + CLASSIFIER_FOLDER + "/" + clf_filename )
    joblib.dump( persisted_classifier, CLASSIFIER_FOLDER + "/" + clf_filename )
    print( "--------------------------" )
    
    if ( not isinstance(classifier, MLPClassifier ) ):
        print( "Predicting recognition accuracy using cross validation...", end="\r" )
        scores = cross_validation( classifier, dataX, dataY )
        print( "Accuracy: %0.4f (+/- %0.4f)                               " % (scores.mean(), scores.std() * 2))
    
    detailed_analysis = input("Should we do a detailed analysis of the model? Y/n" ).lower() == 'y'
    if( detailed_analysis ):
        create_confusion_matrix( classifier, dataX, dataY, directory_names )
        print( "--------------------------" )
    
def determine_labels( dir_path ):
    data_directory_names =  [directory for directory in os.listdir( dir_path ) if not directory.startswith(".")]
    
    print( "Selecting categories to train on... ( [Y]es / [N]o / [S]kip )" )
    filtered_data_directory_names = []
    for directory_name in data_directory_names:
        add = input(" - " + directory_name)
        if( add == "" or add.strip().lower() == "y" ):
            filtered_data_directory_names.append( directory_name )
        elif( add.strip().lower() == "s" ):
            break
        else:
            print( "Disabled " + directory_name )

    return filtered_data_directory_names
