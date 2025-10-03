# io.py
from cobra.io import load_json_model, load_matlab_model, read_sbml_model, save_json_model
from contextlib import redirect_stdout, redirect_stderr
import re
import pandas as pd
import json, os
from typing import Dict, Any
from .ec_utils import convert_to_irreversible, isoenzyme_split

def load_model(model_file):
    try:
        with open(os.devnull, 'w') as f_null:
            with redirect_stdout(f_null), redirect_stderr(f_null):
                if re.search('xml', model_file):
                    model = read_sbml_model(model_file)
                elif re.search('json', model_file):
                    model = load_json_model(model_file)
                elif re.search('mat', model_file):
                    model = load_matlab_model(model_file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_bigg_metabolites(bigg_file):
    return pd.read_csv(bigg_file, sep="\t")

def json_load(path: str) -> Dict[Any, Any]:
    """Loads the given JSON file and returns it as dictionary.

    Arguments
    ----------
    * path: str ~ The path of the JSON file
    """
    with open(path) as f:
        dictionary = json.load(f)
    return dictionary

def json_write(path, dictionary):
    """Writes a JSON file at the given path with the given dictionary as content.

    Arguments
    ----------
    * path:   The path of the JSON file that shall be written
    * dictionary: The dictionary which shalll be the content of
      the created JSON file
    """
    json_output = json.dumps(dictionary, indent=4)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_output)

def trans_model2enz_json_model_split_isoenzyme(model_file, reaction_kcat_mw_file, f, ptot, sigma, lowerbound, upperbound, json_output_file):
    """Tansform cobra model to json mode with  
    enzyme concentration constraintat.

    Arguments
    ----------
    * model_file:   The path of sbml model
    * reaction_kcat_mw_file: The path of storing kcat/MW value of the enzyme catalyzing each
     reaction in the GEM model
    * f: The enzyme mass fraction 
    * ptot: The total protein fraction in cell.  
    * sigma: The approximated average saturation of enzyme. 
    * lowerbound:  Lowerbound  of enzyme concentration constraint. 
    * upperbound:  Upperbound  of enzyme concentration constraint. 

    """
    model = load_model(model_file)
    convert_to_irreversible(model)
    model = isoenzyme_split(model)
    # ./ec_result/citang_irr_enz_constraint.json
    save_folder = os.path.dirname(json_output_file)
    json_path = f"{save_folder}/irreversible.json"
    print('json_path:',json_path)
    save_json_model(model, json_path)
    dictionary_model = json_load(json_path)
    dictionary_model['enzyme_constraint'] = {'enzyme_mass_fraction': f, 'total_protein_fraction': ptot,
                                             'average_saturation': sigma, 'lowerbound': lowerbound, 'upperbound': upperbound}
    # Reaction-kcat_mw file.
    # eg. AADDGT,49389.2889,40.6396,1215.299582180927
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    for eachreaction in range(len(dictionary_model['reactions'])):
        reaction_id = dictionary_model['reactions'][eachreaction]['id']
        #if re.search('_num',reaction_id):
        #    reaction_id=reaction_id.split('_num')[0]
        if reaction_id in reaction_kcat_mw.index:
            dictionary_model['reactions'][eachreaction]['kcat'] = reaction_kcat_mw.loc[reaction_id, 'kcat']
            dictionary_model['reactions'][eachreaction]['kcat_MW'] = reaction_kcat_mw.loc[reaction_id, 'kcat_MW']
        else:
            dictionary_model['reactions'][eachreaction]['kcat'] = ''
            dictionary_model['reactions'][eachreaction]['kcat_MW'] = ''
    json_write(json_output_file, dictionary_model)

def create_file(store_path):
    """
    Create a directory at the specified path.

    Args:
        store_path (str): The path of the directory to create.

    """
    if os.path.exists(store_path):
        print("Path exists")
        # Perform any necessary actions if the path already exists
        # For example, remove the existing directory and create a new one
        # shutil.rmtree(store_path)
        # os.makedirs(store_path)
    else:
        os.makedirs(store_path)
        print(store_path)