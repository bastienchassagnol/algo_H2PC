# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:06:56 2019

@author: Bastien
"""
import time
import pickle

def compute_time(fonction_a_executer):
    def fonction_modifiee(*args, **kwargs):
        """Fonction renvoyée par notre décorateur. Elle se charge
        de calculer le temps mis par la fonction à s'exécuter"""
        
        tps_avant = time.time() # Avant d'exécuter la fonction
        fonction_a_executer(*args, **kwargs) # On exécute la fonction
        return (time.time()-tps_avant)
    return fonction_modifiee

def format_test(test):
    if test[2]:
        return ("{} indep {} given {} ?".format(*test))
    else:
        return ("{} indep {} ?".format(*test[0:2]))
    
def format_args_indep(dico_indep):
    use_pyagrum,use_R,type_test,dof_adjustement=dico_indep.get('usePyAgrum',False),dico_indep.get("R_test",False),dico_indep.get("calculation_method","pearson"),dico_indep.get("dof_adjustment","classic")   
    if use_R:
        return ("{} test using {} dof_adjustement, with R computation".format(type_test,dof_adjustement ))
    elif use_pyagrum:
        return ("{} test using {} dof_adjustement, with Pyagrum computation".format(type_test,dof_adjustement ))
    else:
        return ("{} test using {} dof_adjustement, with own library".format(type_test,dof_adjustement ))
    
def load_objects(file_name):
    with open(file_name, 'rb') as fichier:
         mon_depickler = pickle.Unpickler(fichier)
         return mon_depickler.load()
    
def save_objects(file_name,object_to_save):
    with open(file_name, 'wb') as fichier:
        mon_pickler = pickle.Pickler(fichier)
        mon_pickler.dump(object_to_save)