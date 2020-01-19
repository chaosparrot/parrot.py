from config.config import *
import joblib
from lib.hierarchial_classifier import *
from lib.markov_chain_classifier import *
from lib.change_resistant_classifier import *
from lib.ensemble_classifier import *
import os

def combine_models():
    print( "-------------------------" )
    print( "This script is going to create models using existing models" )
    print( "To increase the likelihood of being good at sound prediction" )
    print( "-------------------------" )

    available_state_dicts = []
    available_models = []
    for fileindex, file in enumerate(os.listdir( CLASSIFIER_FOLDER )):
        if ( file.endswith(".pkl") ):
            available_models.append( file )
        if ( file.endswith(".pth.tar") ):
            available_state_dicts.append( file )            
            
    if( len( available_models ) < 2 ):
        print( "It looks like you haven't trained more than one model yet..." )
        print( "Please train an algorithm first using the [L] option in the main menu" )
        return
        
    print( "What kind of model do you want to make? " )
    print( "[E]nsemble ( default )" )
    print( "[ET]- Ensemble for Pytorch" )
    print( "[H]ierarchial" )
    print( "[C]hange resistant" )
    print( "[M]arkov chain" )
    model_type = input("")
    if( model_type == "" or model_type.strip().lower() == "e" ):
        model_type = "ensemble"
    elif( model_type == "" or model_type.strip().lower() == "et" ):
        model_type = "ensemble_torch"        
    elif( model_type.strip().lower() == "h" ):
        model_type = "hierarchial"        
    elif( model_type.strip().lower() == "c" ):
        model_type = "change_resistant"
    else:
        model_type = "markov_chain"

    default_clf_filename = determine_default_model_name( model_type )
    clf_filename = input("Insert the model name ( empty is '" + default_clf_filename + "' ) ")
    if( clf_filename == "" ):
        clf_filename = default_clf_filename
    clf_filename += '.pkl'
    
    if( model_type == "change_resistant" ):
        classifier_map = configure_base_model( available_models)
    elif( model_type == "markov_chain" or model_type == "hierarchial" ):
        classifier_map = configure_tree_model( available_models )
    elif( model_type == "ensemble_torch"):
        classifier_map = configure_single_layer_model( available_state_dicts, True )   
    else:
        classifier_map = configure_single_layer_model( available_models, False )
            
    connect_model( clf_filename, classifier_map, model_type )
    
def configure_base_model( available_models, text=None ):
    if( text == None ):
        text = "Type the number of the model that you want to make resistant to changes: "

    print_available_models( available_models )
    classifier_file_index = input(text)
    while( int( classifier_file_index ) <= 0 ):
        classifier_file_index = input("")
        if( classifier_file_index.lower() == "x" ):
            return
    classifier_file_index = int( classifier_file_index ) - 1

    main_classifier_file = CLASSIFIER_FOLDER + "/" + available_models[ classifier_file_index ]
    main_classifier = joblib.load( main_classifier_file )
    return {'main': main_classifier}
    
def configure_tree_model( available_models ):
    classifier_map = configure_base_model( available_models, "Type the number of the model that you want to use as a base decision layer: ")
    main_classifier = classifier_map['main']
    labels = main_classifier.classes_
    
    print( "-------------------------" )
    print( "Determining decision layers... " )
    for index, label in enumerate( labels ):
        answer = input( "Should '" + label + "' connect to another model? Y/N ( Empty is no ) " )
        if( answer.lower() == "y" ):
            print_available_models( available_models )
            
            leaf_index = input("Type the number of the model that you want to run when '" + label + "' is detected: ")
            while( int( leaf_index ) <= 0 ):
                leaf_index = input("")
                if( leaf_index.lower() == "x" ):
                    return
            leaf_index = int( leaf_index ) - 1
            classifier_file = CLASSIFIER_FOLDER + "/" + available_models[ leaf_index ]
            classifier_map[ label ] = joblib.load( classifier_file )
    
    return classifier_map
    
def configure_single_layer_model( available_models, pytorch ):
    amount_of_classifiers = input("How many classifiers do you want to add to this model?")
    while( int( amount_of_classifiers ) <= 0 ):
        amount_of_classifiers = input("")
        if( amount_of_classifiers.lower() == "x" ):
            return
    
    classifier_map = {}
    print_available_models( available_models )
    for i in range( int(amount_of_classifiers) ):
        leaf_index = input("Type the number of the model that you want to add: ")
        while( int( leaf_index ) <= 0 ):
            leaf_index = input("")
            if( leaf_index.lower() == "x" ):
                return
        leaf_index = int( leaf_index ) - 1
        classifier_file = CLASSIFIER_FOLDER + "/" + available_models[ leaf_index ]

        if( pytorch ):
            classifier_map[ "classifier_" + str(i) ] = classifier_file
        else:
            classifier_map[ "classifier_" + str(i) ] = joblib.load( classifier_file )
        print( "Added model " + available_models[ leaf_index ] )
        
    return classifier_map
    
def determine_default_model_name( model_type ):
    default_clf_filename = DEFAULT_CLF_FILE + "_combined"
    if( model_type == "markov_chain" ):
        default_clf_filename = DEFAULT_CLF_FILE + "_markov"
    elif( model_type == "change_resistant" ):
        default_clf_filename = DEFAULT_CLF_FILE
    
    return default_clf_filename

def print_available_models( available_models ):
    print( "Available models:" )
    for modelindex, available_model in enumerate(available_models):
        print( " - [" + str( modelindex + 1 ) + "] " + available_model )

def connect_model( clf_filename, classifier_map, model_type ):
    if( model_type == "hierarchial" ):
        classifier = HierarchialClassifier( classifier_map )
    elif( model_type == "markov_chain" ):
        classifier = MarkovChainClassifier( classifier_map )
    elif( model_type == "ensemble" ):
        classifier = EnsembleClassifier( classifier_map )
    elif( model_type == "ensemble_torch" ):
        from lib.torch_ensemble_classifier import TorchEnsembleClassifier    
    
        classifier = TorchEnsembleClassifier( classifier_map )        
    else:
        classifier = ChangeResistantClassifier( classifier_map )
        
    classifier_filename = CLASSIFIER_FOLDER + "/" + clf_filename
    joblib.dump( classifier, classifier_filename )
    print( "-------------------------" )
    print( "Model created!" )
    print( "The created model contains the following " + str(len( classifier.classes_ )) + " possible predictions: " )
    print( ", ".join( classifier.classes_ ) )
    print( "-------------------------" )

