# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:40:17 2019

@author: bchassagno
"""

import pyAgrum as gum
import pandas as pd
import os
from hpc import hpc
import itertools 
import pickle
import pprint as pp
import pyAgrum.lib.ipython as gnb
import pyAgrum.lib.bn_vs_bn as comp
from functools import partial

class H2PC ():
    """H2PC is a new hybrid algorithm combining scoring and constraint-structured learning,
    which can be considered as an improvement of MMHC in many regards. Especially, it clearly enables
    to reduce the number of false negative edges.   
    
    
    """
   
    def __init__(self,df,threshold_pvalue=0.05,verbosity=False,score_algorithm="Greedy_climbing",optimized=False,filtering="AND",**independance_args):
        #check if file is present, if instance of the parameter is correct and the file's extension
        """
        if not isinstance(filename, str):
            raise TypeError("le format attendu pour le fichier d'entrée est de type string")   
        
        if not os.path.isfile(filename):
            raise FileNotFoundError("fichier non trouve a l'emplacement attendu")
        _, extension=os.path.splitext(self.filename)
        
        if extension!=".csv":
            raise TypeError("le format attendu pour le fichier d'entrée est de type .csv")
       
        """
        
        #check non empty values in df
        if not isinstance(df,pd.core.frame.DataFrame):
           raise TypeError ("expected format is a dataframe")
        else:
            if df.isnull().values.any():
                raise ValueError ("we can't perform tests on databases with missng values")
            else:
                self.df=df  
        
        
        if isinstance (threshold_pvalue,numbers.Number): 
            if threshold_pvalue>=0.0 and threshold_pvalue<=1.0:
                self.threshold_pvalue=threshold_pvalue
            else:
                raise ValueError ("Probability must be in range [0,1]")
        else:
            raise TypeError("Pvalue must be a number")        
       
        self.variables=set(df.columns)
        if isinstance(verbosity,bool):
            self.verbosity=verbosity
        else:
            raise TypeError("Expect a boolean for verbosity")
            
        if score_algorithm in ["Greedy_climbing","tabu_search"]:
            self.score_algorithm=score_algorithm
        else:
            raise AssertionError("Only the two following algorithms are for instance suitable : Greedy_climbing, tabu_search")
        
        #neighbourd to check or and and condition
        self.consistent_neighbourhood={}
        
        if isinstance(optimized,bool):
            self.optimized=optimized
        else:
            raise TypeError("Format expected for optimized is boolean")
            
        if filtering in ["AND","OR"]:
            self.filtering=filtering
        else:
            raise AssertionError("Only two filters possible : AND, OR")        
        self.blacklisted=set()
        self.whitelisted=set()
        
        if (score_algorithm=='tabu_search'):
            self.tabu_size,self.nb_decrease=independance_args.get("tabu_size",100),independance_args.get("nb_decrease",50)
        
        self.independance_args=independance_args
        
            
      
    def addForbiddenArc(self,arc):
        if isinstance(arc,gum.pyAgrum.Arc):
            #convert arc into hashable type
            self.blacklisted.add((learner.nameFromId(arc.tail()),learner.nameFromId(arc.head())))
        else:
            raise TypeError("Format expected for learning is pyAgrum.Arc")
    def addMandatoryArc(self,arc):
        if isinstance(arc,gum.pyAgrum.Arc):
            self.whitelisted.add((learner.nameFromId(arc.tail()),learner.nameFromId(arc.head())))
        else:
            raise TypeError("Format expected for learning is pyAgrum.Arc")
            
    def eraseForbiddenArc(self,arc):
        if isinstance(arc,gum.pyAgrum.Arc):
            arc_hashed=(learner.nameFromId(arc.tail()),learner.nameFromId(arc.head()))
            if arc_hashed in self.blacklisted:
                self.blacklisted.remove(arc_hashed)
            else:
                print("Arc '{}' wasn't present in the set of forbidden arcs".format(arc))
        else:
            raise TypeError("Format expected for learning is pyAgrum.Arc")
        
    def eraseMandatoryArc(self,arc):
        if isinstance(arc,gum.pyAgrum.Arc):
            arc_hashed=(arc.tail(),arc.head())
            if arc_hashed in self.blacklisted:
                self.whitelisted.remove(arc_hashed)
            else:
                print("Arc '{}' wasn't present in the set of mandatory arcs".format(arc))
        else:
            raise TypeError("Format expected for learning is pyAgrum.Arc")
    def erase_all_constrainsts(self):
        self.blacklisted,self.whitelisted=set(),set()
        
        
        
    def check_consistency(self,dictionnary_neighbourhood):
        #initialize dictionnary of empty sets
        consistent_dictionnary_neighbourhood={k: set() for k in self.variables}
    
        for couple in itertools.combinations(dictionnary_neighbourhood.keys(),2): 
            
            variable_1,variable_2=couple 
            neighbourhood_variable1=dictionnary_neighbourhood[variable_1].copy()
            neighbourhood_variable2=dictionnary_neighbourhood[variable_2].copy()
            if self.filtering=="AND":
                if (variable_1 in neighbourhood_variable2) and (variable_2 in neighbourhood_variable1):                
                    #under the assumption of exctness of tests, if variable 1 is in neighbourhood of variable 2
                    #is equivalent that variable 2 is in neighbourhood of variable 1
                    consistent_dictionnary_neighbourhood[variable_1]=consistent_dictionnary_neighbourhood[variable_1].union({variable_2})
                    consistent_dictionnary_neighbourhood[variable_2]=consistent_dictionnary_neighbourhood[variable_2].union({variable_1})
            else:
                if (variable_1 in neighbourhood_variable2) or (variable_2 in neighbourhood_variable1):                
                    #under the assumption of exctness of tests, if variable 1 is in neighbourhood of variable 2
                    #is equivalent that variable 2 is in neighbourhood of variable 1
                    consistent_dictionnary_neighbourhood[variable_1]=consistent_dictionnary_neighbourhood[variable_1].union({variable_2})
                    consistent_dictionnary_neighbourhood[variable_2]=consistent_dictionnary_neighbourhood[variable_2].union({variable_1})
                
             
        return consistent_dictionnary_neighbourhood
       
        
    def _unique_edges(self,consistent_dictionnary):
        set_unique_edges=set()
        for variable in consistent_dictionnary.keys():
            #to check if neighbourhood is not empty
            if consistent_dictionnary[variable]:               
                for neighbour in consistent_dictionnary[variable]:
                    edge=(variable,neighbour)
                    set_unique_edges.add(edge)
        return (set_unique_edges)
     
    def _add_set_unique_possible_edges(self,unique_possible_edges):
        for unique_edge in unique_possible_edges:
            self.learner.addPossibleEdge(*unique_edge)
           
     
    def _learn_graphical_structure(self):
        possible_algorithm = {'Greedy_climbing': self.learner.useGreedyHillClimbing, 'tabu_search': self.learner.useLocalSearchWithTabuList}        
        
        possible_algorithm[self.score_algorithm](*self.independance_args)
        bn_learned=self.learner.learnBN()  
        return bn_learned
    
    
    def _HPC_global(self):
        dico_couverture_markov={}
        for target in self.variables:  
            print("la variable est ",target)
            dico_couverture_markov[target]=hpc(target,self.df,self.threshold_pvalue,self.verbosity,whitelisted=self.whitelisted,blacklisted=self.blacklisted,independance_args=self.independance_args).couverture_markov()           
            print("We compute with HPC the neighbours of '{}' : '{}' \n\n".format(target,dico_couverture_markov[target]))
            if self.verbosity:
                print("We compute with HPC the neighbours of '{}' : '{}' \n\n".format(target,dico_couverture_markov[target]))
        return dico_couverture_markov
        
    def _HPC_optimized(self):
      
        dico_couverture_markov={}
        known_bad,known_good=set(),set()
        for target in self.variables:
            #check if a part of the dictionnary was computed or if it is still empty
            
            if dico_couverture_markov:             
                   
                known_bad={kv[0] for kv in dico_couverture_markov.items() if target not in kv[1]}
                known_good={kv[0] for kv in dico_couverture_markov.items() if target in kv[1]} 
                if self.verbosity:
                    print("known good nodes inferred are '{}' and known bad nodes inferred are '{}' ".format(known_bad,known_good))                    
            dico_couverture_markov[target]=hpc(target,self.df,self.threshold_pvalue,self.verbosity,whitelisted=self.whitelisted,blacklisted=self.blacklisted,known_bad=known_bad,known_good=known_good,independance_args=self.independance_args).couverture_markov()["neighbours"]
            if self.verbosity:
                print("We compute with HPC the neighbours of '{}' :'{}' \n\n".format(target,dico_couverture_markov[target]) )
            print("We compute with HPC the neighbours of '{}' :'{}' \n\n".format(target,dico_couverture_markov[target]) )
        return dico_couverture_markov
                
                    
                    
    
    
    def learnBN(self):
        #computation of local neighbourhood for each node 
   
        if self.optimized:
            dico_couverture_markov=self._HPC_optimized()
        else:
            dico_couverture_markov=self._HPC_global()      
        
        self.consistent_neighbourhood=self.check_consistency(dico_couverture_markov)
        """
        with open('dictionnary', 'wb') as fichier:
             mon_pickler = pickle.Pickler(fichier)
             mon_pickler.dump(self.consistent_neighbourhood)
        
      
        with open('dictionnary', 'rb') as fichier:
            mon_depickler = pickle.Unpickler(fichier)
            dico_couverture_markov = mon_depickler.load()
        #print("le dico apres verification consistence est ",pp.pprint(consistent_dictionnary,width=1))    
        """
       
        unique_possible_edges=self._unique_edges(self.consistent_neighbourhood)
        #add set of unique edges as unique possible addings for h2pc
        #print("set of unique possible edges is ",unique_possible_edges)
        if self.verbosity:
            print("set of unique possible edges is '{}'".format(unique_possible_edges))
        #score_based learning according to input_score
        self._add_set_unique_possible_edges(unique_possible_edges)   
        bn_learned=self._learn_graphical_structure()    
        return bn_learned
  
   
        
    
        
        
        
        
        
        
if __name__ == "__main__":
    true_bn=gum.loadBN(os.path.join("true_graphes_structures","alarm.bif"))
    #gnb.showBN(true_bn,size="4")
    gum.generateCSV(true_bn,"sample_alarm.csv",20000,False)
    
    #with tabu search
    
    #learner=gum.BNLearner("sample_alarm.csv") 
    """
    learner.useLocalSearchWithTabuList()
    bn2=learner.learnBN()    
    gnb.showBN(bn2,size="4")
    """
    
    
    
    #with h2pc coupled with tabu search
    df=pd.read_csv("sample_asia.csv")
    objet_1=H2PC(df,threshold_pvalue=0.05,verbosity=False,score_algorithm="Greedy_climbing",optimized=False,filtering="AND",dof_adjustment="classic").learnBN()
    #dag_h2pc=H2PC(learner,score_algorithm="tabu_search",optimized=False,filtering="OR",tabu_size=1000,nb_decrease=500).learnBN()
    #gnb.showBN(dag_h2pc,size="4")
    
    
    



    
    
    
    
  
    
  

    
    
    
    

   
    
   
    
 

    

    
 
    
    
    
        
            
        
        
    
    
    