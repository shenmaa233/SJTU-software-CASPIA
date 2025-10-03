import re
import pandas as pd
from cobra.core import Reaction
from cobra.util.solver import set_objective
from cobra.io import load_json_model
import json
from typing import Dict, Any

def get_currency_metabolites():
    '''
    Returns a set of currency metabolites.
    '''
    currencylist1 = ['coa_c', 'co2_c', 'co2_e', 'co2_p', 'cobalt2_c', 'cobalt2_e', 'cobalt2_p', 'h_c', 'h_e', 'h_p', 'h2_c', 'h2_e', 'h2_p', 'h2o_c', 'h2o_e', 'h2o_p', 'h2o2_c', 'h2o2_e', 'h2o2_p', 'nh4_c', 'nh4_e', 'nh4_p', 'o2_c', 'o2_e', 'o2_p', 'pi_c', 'pi_e', 'pi_p', 'ppi_c', 'pppi_c', 'q8_c', 'q8h2_c', 'no_p', 'no3_p', 'no3_e', 'no_c', 'no2_c', 'no_e', 'no3_c', 'no2_e', 'no2_p', 'n2o_c', 'n2o_e', 'n2o_p', 'h2s_c', 'so3_c', 'so3_p', 'o2s_c', 'h2s_p', 'so2_e', 'so4_e', 'h2s_e', 'o2s_p', 'o2s_e', 'so4_c', 'so4_p', 'so3_e', 'so2_c', 'so2_p', 'ag_c', 'ag_e', 'ag_p','na1_c','na1_e','na1_p','ca2_c', 'ca2_e', 'ca2_p', 'cl_c', 'cl_e', 'cl_p', 'cd2_c', 'cd2_e', 'cd2_p', 'cu_c', 'cu_e', 'cu_p', 'cu2_c', 'cu2_e', 'cu2_p', 'fe2_c', 'fe2_e', 'fe2_p', 'fe3_c', 'fe3_e', 'fe3_p', 'hg2_c', 'hg2_e', 'hg2_p', 'k_c', 'k_e', 'k_p', 'mg2_c', 'mg2_e', 'mg2_p', 'mn2_c', 'mn2_e', 'mn2_p', 'zn2_c', 'zn2_e', 'zn2_p','nh3']
    currencylist2 = ['amp_c', 'amp_e', 'amp_p', 'adp_c', 'adp_e', 'adp_p', 'atp_c', 'atp_e', 'atp_p', 'cmp_c', 'cmp_e', 'cmp_p', 'cdp_c', 'cdp_e', 'cdp_p', 'ctp_c', 'ctp_e', 'ctp_p', 'gmp_c', 'gmp_e', 'gmp_p', 'gdp_c', 'gdp_e', 'gdp_p', 'gtp_c', 'gtp_e', 'gtp_p', 'imp_c', 'imp_e', 'imp_p', 'idp_c', 'idp_e', 'idp_p', 'itp_c', 'itp_e', 'itp_p', 'ump_c', 'ump_e', 'ump_p', 'udp_c', 'udp_e', 'udp_p', 'utp_c', 'utp_e', 'utp_p', 'xmp_e', 'xmp_c', 'xmp_p', 'xdp_c', 'xdp_e', 'xdp_p', 'xtp_c', 'xtp_e', 'xtp_p', 'damp_c', 'damp_e', 'damp_p', 'dadp_c', 'dadp_e', 'dadp_p', 'datp_c', 'datp_e', 'datp_p', 'dcmp_c', 'dcmp_e', 'dcmp_p', 'dcdp_c', 'dcdp_e', 'dcdp_p', 'dctp_c', 'dctp_e', 'dctp_p', 'dgmp_c', 'dgmp_e', 'dgmp_p', 'dgdp_c', 'dgdp_e', 'dgdp_p', 'dgtp_c', 'dgtp_e', 'dgtp_p', 'dimp_c', 'dimp_e', 'dimp_p', 'didp_c', 'didp_e', 'didp_p', 'ditp_c', 'ditp_e', 'ditp_p', 'dump_c', 'dump_e', 'dump_p', 'dudp_c', 'dudp_e', 'dudp_p', 'dutp_c', 'dutp_e', 'dutp_p', 'dtmp_c', 'dtmp_e', 'dtmp_p', 'dtdp_c', 'dtdp_e', 'dtdp_p', 'dttp_c', 'dttp_e', 'dttp_c', 'fad_c', 'fad_p', 'fad_e', 'fadh2_c', 'fadh2_e', 'fadh2_p', 'nad_c', 'nad_e', 'nad_p', 'nadh_c', 'nadh_e', 'nadh_p', 'nadp_c', 'nadp_e', 'nadp_p', 'nadph_c', 'nadph_e', 'nadph_p']
    currencylist3 = ['cdp', 'ag', 'dctp', 'dutp', 'ctp', 'gdp', 'gtp', 'ump', 'ca2', 'h2o', 'datp', 'co2', 'no2', 'no', 'k', 'zn2', 'no3', 'o2', 'cl', 'udp', 'damp', 'ditp', 'dump', 'q8h2', 'pppi', 'idp', 'dimp', 'pi', 'dttp', 'so4', 'adp', 'xtp', 'dgtp', 'dadp', 'coa', 'ppi', 'h2', 'cmp', 'fe2', 'o2s', 'h', 'gmp', 'itp', 'q8', 'cobalt2', 'n2o', 'xmp', 'xdp', 'nadph', 'cu', 'cu2', 'atp', 'dgmp', 'imp', 'h2s', 'utp', 'dtmp', 'fadh2', 'so3', 'fad', 'cd2', 'dgdp', 'nad', 'nadh', 'hg2', 'dcmp', 'dudp', 'dtdp', 'didp', 'mn2', 'dcdp', 'nh4', 'amp', 'fe3', 'nadp', 'so2', 'h2o2', 'mg2']
    return set(currencylist1 + currencylist2 + currencylist3)

def extract_metabolite_id(met_id):
    '''
    Extracts the metabolite ID without compartment information.
    '''
    if '_c' in met_id:
        return met_id.split('_c')[0]
    elif '_p' in met_id:
        return met_id.split('_p')[0]
    elif '_e' in met_id:
        return met_id.split('_e')[0]
    return met_id

def isoenzyme_split(model):
    """Split isoenzyme reaction to mutiple reaction

    Arguments
    ----------
    * model: cobra.Model.
    
    :return: new cobra.Model.
    """  
    for r in model.reactions:
        if re.search(" or ", r.gene_reaction_rule):
            rea = r.copy()
            gene = r.gene_reaction_rule.split(" or ")
            for index, value in enumerate(gene):
                if index == 0:
                    r.id = r.id + "_num1"
                    r.gene_reaction_rule = value
                else:
                    r_add = rea.copy()
                    r_add.id = rea.id + "_num" + str(index+1)
                    r_add.gene_reaction_rule = value
                    #model.add_reaction(r_add)#3.7
                    model.add_reactions([r_add])#3.8
    for r in model.reactions:
        r.gene_reaction_rule = r.gene_reaction_rule.strip("( )")
    return model

def convert_to_irreversible(model):
    """Split reversible reactions into two irreversible reactions

    These two reactions will proceed in opposite directions. This
    guarentees that all reactions in the model will only allow
    positive flux values, which is useful for some modeling problems.

    Arguments
    ----------
    * model: cobra.Model ~ A Model object which will be modified in place.

    """
    #warn("deprecated, not applicable for optlang solvers", DeprecationWarning)
    reactions_to_add = []
    coefficients = {}
    for reaction in model.reactions:
        if reaction.lower_bound < 0 and reaction.upper_bound == 0:
            for metabolite in reaction.metabolites:
                original_coefficient = reaction.get_coefficient(metabolite)
                reaction.add_metabolites({metabolite: -2*original_coefficient})
            reaction.id += "_reverse"
            reaction.upper_bound = -reaction.lower_bound
            reaction.lower_bound = 0
        # If a reaction is reverse only, the forward reaction (which
        # will be constrained to 0) will be left in the model.
        if reaction.lower_bound < 0 and reaction.upper_bound > 0:
            reverse_reaction = Reaction(reaction.id + "_reverse")
            reverse_reaction.lower_bound = max(0, -reaction.upper_bound)
            reverse_reaction.upper_bound = -reaction.lower_bound
            coefficients[
                reverse_reaction] = reaction.objective_coefficient * -1
            reaction.lower_bound = max(0, reaction.lower_bound)
            reaction.upper_bound = max(0, reaction.upper_bound)
            # Make the directions aware of each other
            reaction.notes["reflection"] = reverse_reaction.id
            reverse_reaction.notes["reflection"] = reaction.id
            reaction_dict = {k: v * -1
                             for k, v in reaction._metabolites.items()}
            reverse_reaction.add_metabolites(reaction_dict)
            reverse_reaction._model = reaction._model
            reverse_reaction._genes = reaction._genes
            for gene in reaction._genes:
                gene._reaction.add(reverse_reaction)
            reverse_reaction.subsystem = reaction.subsystem
            reverse_reaction.gene_reaction_rule = reaction.gene_reaction_rule
            try:  
                reaction.annotation
            except:
                pass
            else:
                reverse_reaction.annotation = reaction.annotation
            reactions_to_add.append(reverse_reaction)
    model.add_reactions(reactions_to_add)
    set_objective(model, coefficients, additive=True)

def json_load(path: str) -> Dict[Any, Any]:
    """Loads the given JSON file and returns it as dictionary.

    Arguments
    ----------
    * path: str ~ The path of the JSON file
    """
    with open(path) as f:
        dictionary = json.load(f)
    return dictionary

def preprocess_metabolites(model, currency_metabolites):
    '''
    Preprocesses the metabolites and creates a DataFrame.

    Arguments
    ----------
    * model: cobra.Model ~ A Model object to extract metabolite information from.
    * currency_metabolites: set ~ A set of currency metabolite IDs.

    Returns
    ----------
    * gprdf: pandas.DataFrame ~ DataFrame with preprocessed metabolite information.
    '''
    rlist, sublist, subtotal, gprlist, genetotal = [], [], [], [], []

    for reaction in model.reactions:
        if reaction.gene_reaction_rule and len(reaction.reactants) > 1:
            for metabolite in reaction.reactants:
                if metabolite.id not in currency_metabolites:
                    for gene in reaction.genes:
                        rlist.append(reaction.id)
                        sublist.append(metabolite.id)
                        gprlist.append(reaction.gene_reaction_rule)
                        genetotal.append(gene.id)
                        subtotal.append(extract_metabolite_id(metabolite.id))

    gprdf = pd.DataFrame({
        'reactions': rlist,
        'metabolites': sublist,
        'metabolitestotal': subtotal,
        'gpr': gprlist,
        'genes': genetotal,
    })

    return gprdf


    
def get_enzyme_constraint_model(json_model_file):
    """using enzyme concentration constraint
    json model to create a COBRApy model.

    Arguments
    ----------
    * json_model_file: json Model file.

    :return: Construct an enzyme-constrained model.
    """

    dictionary_model = json_load(json_model_file)
    model = load_json_model(json_model_file)

    coefficients = dict()
    for rxn in model.reactions:
        for eachr in dictionary_model['reactions']:
            if rxn.id == eachr['id']:
                if eachr['kcat_MW']:
                    coefficients[rxn.forward_variable] = 1 / float(eachr['kcat_MW'])
                break

    lowerbound = dictionary_model['enzyme_constraint']['lowerbound']
    upperbound = dictionary_model['enzyme_constraint']['upperbound']
    constraint = model.problem.Constraint(0, lb=lowerbound, ub=upperbound)
    model.add_cons_vars(constraint)
    model.solver.update()
    constraint.set_linear_coefficients(coefficients=coefficients)
    return model

def get_model_substrate_obj(use_model):
    '''
    change model substrate for single carbon source
    
    Arguments
    ----------
    use_model: cobra.Model ~ A Model object which will be modified in place.
    '''
    
    ATPM='No' 
    substrate_list=[]
    concentration_list=[]
    EX_exclude_reaction_list=['EX_pi_e','EX_h_e','EX_fe3_e','EX_mn2_e','EX_co2_e','EX_fe2_e','EX_h2_e','EX_zn2_e',\
                             'EX_mg2_e','EX_ca2_e','EX_so3_e','EX_ni2_e','EX_no_e','EX_cu2_e','EX_hg2_e','EX_cd2_e',\
                             'EX_h2o2_e','EX_h2o_e','EX_no2_e','EX_nh4_e','EX_so4_e','EX_k_e','EX_na1_e','EX_o2_e',\
                             'EX_o2s_e','EX_ag_e','EX_cu_e','EX_so2_e','EX_cl_e','EX_n2o_e','EX_cs1_e','EX_cobalt2_e']
    EX_exclude_reaction_list=EX_exclude_reaction_list+[i+'_reverse' for i in EX_exclude_reaction_list]
    for r in use_model.reactions:
        if r.objective_coefficient == 1:
            obj=r.id #Product name
        #elif not r.lower_bound==0 and not r.lower_bound==-1000 and not r.lower_bound==-999999 and abs(r.lower_bound)>0.1:#排除很小的值
        elif not r.upper_bound==0 and not r.upper_bound==1000 and not r.upper_bound==999999 and abs(r.upper_bound)>0.1:#排除很小的值
            #print(r.id,r.upper_bound,r.lower_bound)
            if r.id=='ATPM':
                if r.upper_bound>0:
                    ATPM='Yes' #ATP maintenance requirement
            elif r.id not in EX_exclude_reaction_list:
                #print(r.id,r.upper_bound,r.lower_bound)
                #substrate=r.id #Substrate name
                substrate_list.append(r.id)
                #concentration=r.upper_bound #Substrate uptake rate  
                concentration_list.append(r.upper_bound)
    return(obj,substrate_list,concentration_list,ATPM)


def get_fluxes_detail_in_model(model,model_pfba_solution,fluxes_outfile,json_model_file):
    """
    Get the detailed information of each reaction.

    Arguments:
    * model: cobra.Model - The metabolic model.
    * model_pfba_solution: pandas.Series - The pFBA solution containing reaction fluxes.
    * fluxes_outfile: str - Path to the output file for reaction fluxes.
    * json_model_file: str - Path to the JSON model file.

    Returns:
    * model_pfba_solution_detail: pandas.DataFrame - Detailed information of each reaction.
    """

    dictionary_model = json_load(json_model_file)
    model_pfba_solution = model_pfba_solution.to_frame()
    model_pfba_solution_detail = pd.DataFrame()
    for index, row in model_pfba_solution.iterrows():
        reaction_detail = model.reactions.get_by_id(index)
        model_pfba_solution_detail.loc[index, 'fluxes'] = row['fluxes']
        for eachreaction in dictionary_model['reactions']:
            if index ==eachreaction['id']:
                if 'annotation' in eachreaction.keys():
                    if 'ec-code' in eachreaction['annotation'].keys():
                        if isinstance (eachreaction['annotation']['ec-code'],list):
                            model_pfba_solution_detail.loc[index, 'ec-code'] = (',').join(eachreaction['annotation']['ec-code'])
                        else:
                            model_pfba_solution_detail.loc[index, 'ec-code'] = eachreaction['annotation']['ec-code']    
                if 'kcat_MW' in eachreaction.keys():
                    if eachreaction['kcat_MW']:
                        model_pfba_solution_detail.loc[index, 'kcat_MW'] = eachreaction['kcat_MW']
                        model_pfba_solution_detail.loc[index, 'E'] = float(row['fluxes'])/float(eachreaction['kcat_MW'])
                break
        model_pfba_solution_detail.loc[index, 'equ'] = reaction_detail.reaction
    model_pfba_solution_detail.to_csv(fluxes_outfile)
    return model_pfba_solution_detail
