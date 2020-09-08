#import subprocess
from function import load_pickle, two_fighters_accuracy, test_model_on_selected_photo

def main_runner():
    #subprocess.call(["python", "function.py"])
    
    load_model, load_out_encoder = load_pickle()

    two_fighters_accuracy(load_model, load_out_encoder)

    test_model_on_selected_photo(URL_TEST, load_model, load_out_encoder)

    return

if __name__ == "__main__":
    main_runner()