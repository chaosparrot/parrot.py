from config.config import *
import joblib
from lib.hierarchial_classifier import *
from lib.ensemble_classifier import *
import os
from lib.audio_model import *

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
            
    if( len( available_models ) < 2 and len(available_state_dicts) < 1 ):
        print( "It looks like you haven't trained more than one model yet..." )
        print( "Please train an algorithm first using the [L] option in the main menu" )
        return
        
    print( "What kind of model do you want to make? " )
    print( "[E]nsemble ( default )" )
    if( PYTORCH_AVAILABLE ):
        print( "[ET]- Ensemble for Pytorch" )
    print( "[H]ierarchial" )
    print( "[U]pgrade model to new version, or change audio settings" )
    model_type = input("")
    if( model_type == "" or model_type.strip().lower() == "e" ):
        model_type = "ensemble"
    elif( model_type.strip().lower() == "et" and PYTORCH_AVAILABLE ):
        model_type = "ensemble_torch"
    elif( model_type.strip().lower() == "h" ):
        model_type = "hierarchial"        
    elif( model_type.strip().lower() == "u" ):
        model_type = "upgrade"

    if(model_type == "upgrade"):
        update_model(available_models)
    else:
        default_clf_filename = DEFAULT_CLF_FILE + "_combined"
        clf_filename = input("Insert the model name ( empty is '" + default_clf_filename + "' ) ")
        if( clf_filename == "" ):
            clf_filename = default_clf_filename
        clf_filename += '.pkl'
        
        if( model_type == "hierarchial" ):
            classifier_map = configure_tree_model( available_models )
        elif( model_type == "ensemble_torch"):
            classifier_map = configure_single_layer_model( available_state_dicts, True )
        else:
            classifier_map = configure_single_layer_model( available_models, False )
                
        connect_model( clf_filename, classifier_map, model_type )
    
def configure_base_model( available_models, text=None, with_filename=False ):
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
    
    base_model = {'main': main_classifier}
    if (with_filename):
        base_model['filename'] = main_classifier_file

    return base_model
    
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

def update_model( available_models ):
    classifier_map = configure_base_model( available_models, "Type the number of the model that you want to upgrade: ", True)
    main_classifier = classifier_map['main']
    classifier_filename = classifier_map['filename']

    print( "-------------------------" )     
    if ( not isinstance(main_classifier, AudioModel) ):
        print( "Current version: v0" )
        settings = define_settings( get_current_default_settings() )
        main_classifier = AudioModel(settings, main_classifier)
        joblib.dump( main_classifier, classifier_filename )
    else:
        print( "Current version: v" + str(main_classifier.settings['version']) )    
        main_classifier.settings = define_settings( main_classifier.settings )
        joblib.dump( main_classifier, classifier_filename )
        
        from lib.torch_ensemble_classifier import TorchEnsembleClassifier
        if isinstance(main_classifier.classifier, TorchEnsembleClassifier) and \
            int(main_classifier.settings['FEATURE_ENGINEERING_TYPE']) == 4:
            import torch
            from lib.machinelearning import feature_engineering_raw            
                        
            classifier_filename = classifier_filename.replace(".pkl", ".onnx")
            print("Detected Talon Voice capable model - Exporting "+ classifier_filename + " file - Use this for the parrot.py integration")
            
            audio_input_size = main_classifier.settings['RATE'] * main_classifier.settings['RECORD_SECONDS']
            true_input_size = len(feature_engineering_raw(torch.randn((int(audio_input_size))).numpy(), \
                main_classifier.settings['RATE'], 0, main_classifier.settings['RECORD_SECONDS'], main_classifier.settings['FEATURE_ENGINEERING_TYPE'])[0])
            dummy_input = torch.randn(1, true_input_size, requires_grad=False).double()
                        
            torch_classifier = main_classifier.classifier.combinedClassifier
            torch_classifier.eval()
            
            # The models inside need to be detached from require_grad constant to be exportable
            for model in torch_classifier.models:
                for param in model.parameters():
                    param.requires_grad = False
           
            torch.onnx.export(
                torch_classifier,
                (dummy_input, ),
                classifier_filename,
                opset_version=13,
                export_params=True,
                verbose=False
            )
            
            # For some odd reason importing onnx before this place makes python stall
            import onnx
            
            model = onnx.load(classifier_filename)
            meta = model.metadata_props.add()
            meta.key = "output_labels"
            meta.value = ";".join(main_classifier.classes_)
            onnx.save(model, classifier_filename)
                        

def define_settings(settings):
    print( "Use the current audio settings for this model? Y/N ( Empty is yes )" )
    use_default = input("")
    if (use_default == "n"):
        print("Bitrate: Current " + str(settings['RATE']) )
        rate = input("")
        if (rate != "" and int(rate) > 0 ):
            settings['RATE'] = int(rate)    
    
        print("Amount of audio channels: Current " + str(settings['CHANNELS']) )
        channels = input("")
        if (channels != "" and int(channels) > 0 ):
            settings['CHANNELS'] = int(channels)
            
        print("Sound frame length : Current " + str(settings['RECORD_SECONDS']) + " seconds" )
        record_seconds = input("")
        if (record_seconds != "" and float(record_seconds) > 0 ):
            settings['RECORD_SECONDS'] = float(record_seconds)    

        print("Amount of sliding windows ( split frames ): Current " + str(settings['SLIDING_WINDOW_AMOUNT']))
        sliding_window_amount = input("")
        if (sliding_window_amount != "" and int(sliding_window_amount) > 0 ):
            settings['SLIDING_WINDOW_AMOUNT'] = int(sliding_window_amount)
            
        print("Input type: Current " + str(settings['FEATURE_ENGINEERING_TYPE']))
        print("-------------------------")
        print("Type the number of the input method that you want to use" )
        print("[1] - RAW - Raw WAVE input")        
        print("[2] - V0.8 - MFCC input with frequency and intensity")
        print("[3] - V0.9 - Normalized MFCC input")
        print("[4] - V0.12 - Normalized MFSC input")        
        feature_engineering = input("")
        if (feature_engineering != "" and int(feature_engineering) > 0 and int(feature_engineering) < 5 ):
            settings['FEATURE_ENGINEERING_TYPE'] = int(feature_engineering)
    print( "-------------------------" )
    return settings


def get_current_default_settings():
    return {
        'version': 1,
        'RATE': RATE,
        'CHANNELS': CHANNELS,
        'RECORD_SECONDS': RECORD_SECONDS,
        'SLIDING_WINDOW_AMOUNT': SLIDING_WINDOW_AMOUNT,
        'FEATURE_ENGINEERING_TYPE': FEATURE_ENGINEERING_TYPE
    }

def print_available_models( available_models ):
    print( "Available models:" )
    for modelindex, available_model in enumerate(available_models):
        print( " - [" + str( modelindex + 1 ) + "] " + available_model )

def connect_model( clf_filename, classifier_map, model_type, during_training = False, settings = None ):
    if( model_type == "hierarchial" ):
        classifier = HierarchialClassifier( classifier_map )
    elif( model_type == "ensemble" ):
        classifier = EnsembleClassifier( classifier_map )
    elif( model_type == "ensemble_torch" ):
        from lib.torch_ensemble_classifier import TorchEnsembleClassifier    
        classifier = TorchEnsembleClassifier( classifier_map )

    if (settings == None):
        settings = define_settings( get_current_default_settings() )
    classifier = AudioModel( settings, classifier )
    classifier_filename = CLASSIFIER_FOLDER + "/" + clf_filename
    joblib.dump( classifier, classifier_filename )
    
    if (during_training == False):
        print( "-------------------------" )
        print( "Model created!" )
        print( "The created model contains the following " + str(len( classifier.classes_ )) + " possible predictions: " )
        print( ", ".join( classifier.classes_ ) )
        print( "-------------------------" )

