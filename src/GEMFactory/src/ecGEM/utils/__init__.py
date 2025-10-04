import pandas as pd
from .io_utils import *
from .ec_utils import *
from .metabolite_utils import *
from .protein_utils import *
from .kcat_utils import *
from .parameter_utils import *

# 进行EC算法前的模型适用性判断
# 第二个参数为固定值（bigg_models_metabolites.txt这个文件）
def Determine_suitable_ecGEM(model_file, bigg_met_file):
    '''
    Determine if the provided model is suitable for constructing an enzyme-constrained model.

    Arguments:
    * model_file: str - File path of the model in JSON/MATLAB/SBML format.
    * bigg_met_file: str - File path of the Bigg metabolite data file in tab-separated format.

    Returns:
    * result: (str, list) - A tuple where the first element is 'Yes' or 'No' indicating suitability and the second element is a list of error messages or a success message.
    '''

    error_list = []

    # Read the model from the file
    model = load_model(model_file)
    

    # Load the Bigg metabolite data
    bigg_met_df = pd.read_csv(bigg_met_file, sep='\t')

    # Check metabolite coverage
    model_met_list = [met.id for met in model.metabolites]
    model_met_in_bigg = [met.id for met in model.metabolites if met.id in bigg_met_df['bigg_id'].tolist()]
    met_coverage = len(model_met_in_bigg) / len(model_met_list)
    if met_coverage < 0.25:
        met_error = f"The coverage of metabolites is too low ({met_coverage*100:.1f}%), and it is not recommended to construct an enzyme-constrained model."
        error_list.append(met_error)

    # Check reaction coverage
    model_reaction_list = [reaction.id for reaction in model.reactions if not reaction.id.startswith('EX_')]
    model_reaction_with_EC = [reaction.id for reaction in model.reactions if 'ec-code' in reaction.annotation]
    reaction_coverage = len(model_reaction_with_EC) / len(model_reaction_list)
    if reaction_coverage < 0.25:
        reaction_error = f"The coverage of reactions is too low ({reaction_coverage*100:.1f}%), and it is not recommended to use DLKcat to obtain enzyme kinetic data."
        error_list.append(reaction_error)

    sui_or_not='Yes'
    if len(error_list) > 0:
        sui_or_not='No'
    else:
        error_list.append("Suitable for constructing enzyme-bound models.")

    return (sui_or_not,error_list)

