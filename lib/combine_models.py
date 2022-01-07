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
            
            class M(torch.nn.Module):
                
                def __init__(self):
                    super(M, self).__init__()                    
                    self.fc1 = torch.nn.Linear(45, 45)
                
                def forward(self, x):
                    return self.fc1(x)

            classifier_filename = classifier_filename.replace(".pkl", ".onnx")            
            print("Detected Talon Voice capable model - Exporting "+ classifier_filename + " file - Use this for the parrot.py integration")
            print()
            
            audio_input_size = main_classifier.settings['RATE'] * main_classifier.settings['RECORD_SECONDS']
            true_input_size = len(feature_engineering_raw(torch.randn((int(audio_input_size))).numpy(), \
                main_classifier.settings['RATE'], 0, main_classifier.settings['RECORD_SECONDS'], main_classifier.settings['FEATURE_ENGINEERING_TYPE'])[0])
            input_names = []
            for x in range(true_input_size):
                input_names.append("mfsc" + str(x))
            dummy_input = torch.from_numpy(np.asarray([torch.zeros((true_input_size), requires_grad=False).numpy()])).double()#torch.from_numpy(np.asarray([torch.zeros((true_input_size)).numpy()])).double()
            #dummy_input.requires_grad = False
            dummy_output = torch.zeros((len(main_classifier.classifier.classes_)), requires_grad=False)
            #dummy_output[0] = 1.0
            
#real input - tensor([[ 2.1422,  2.2133,  2.1466,  1.4511,  0.1155, -1.1442, -1.0249, -0.9786,
#         -1.1083, -1.3470, -1.2139, -1.3049, -1.3878, -1.4346, -1.3665, -1.4997,
#         -1.4478, -1.4888, -1.5303, -1.4382, -0.9869, -0.6087, -1.1574, -1.4752,
#         -1.5790, -1.5807, -1.6291, -1.6102, -1.1462, -1.0878, -1.3290, -1.3041,
#         -1.5277, -1.5464, -1.4652, -1.0432, -1.0930, -1.2181, -1.3096, -1.2863,
#          2.5656,  2.6944,  2.7083,  2.1283,  0.3716, -0.1843, -0.1956, -0.2903,
#         -0.4607, -0.4973, -0.4931, -0.5619, -0.5920, -0.6485, -0.6085, -0.6563,
#         -0.6779, -0.6801, -0.6483, -0.6502, -0.6543, -0.6230, -0.6128, -0.6159,
#         -0.5746, -0.5574, -0.5318, -0.4983, -0.4473, -0.4280, -0.4183, -0.3923,
#         -0.3701, -0.3481, -0.3174, -0.2826, -0.2894, -0.2667, -0.2521, -0.4344,
#          2.0556,  2.2507,  2.3801,  2.2066,  1.6269,  1.2333,  1.0383,  0.9455,
#          0.7903,  0.7336,  0.7643,  0.6773,  0.6354,  0.6013,  0.6427,  0.5704,
#          0.5766,  0.5747,  0.5822,  0.6114,  0.5821,  0.6289,  0.6335,  0.6487,
#          0.6766,  0.7033,  0.7261,  0.7549,  0.7863,  0.8132,  0.8454,  0.8644,
#          0.8948,  0.9132,  0.9412,  0.9587,  0.9742,  0.9950,  1.0029,  0.7893,
#          0.3443,  0.4979,  0.6339,  0.7373,  0.7481,  0.6207,  0.3825,  0.0924,
#          0.0639,  0.1453,  0.1341, -0.1327, -0.0860,  0.0302, -0.1119, -0.1549,
#         -0.0161, -0.2056, -0.0416, -0.0974, -0.0857, -0.0602, -0.0189, -0.0633,
#          0.0421,  0.0090,  0.0486,  0.0979,  0.1179,  0.1386,  0.1728,  0.1962,
#          0.2246,  0.2435,  0.2691,  0.2896,  0.3066,  0.3203,  0.3334,  0.1258]],
#       dtype=torch.float64)            
                        
            print( "OUTPUT", dummy_output, main_classifier.classes_ )
            torch_classifier = main_classifier.classifier.combinedClassifier
            torch_classifier.eval()
            
            m = M()
            
            # The models inside need to be detached from require_grad constant to be exportable
            for model in torch_classifier.models:
                for param in model.parameters():
                    param.requires_grad = False
                
            torch.onnx.export(
                torch_classifier,
                dummy_input,
                classifier_filename,
                #input_names=input_names,
                output_names=main_classifier.classes_,
                example_outputs=dummy_output,
                opset_version=13,
                export_params=True,
                verbose=False
            )
        

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

