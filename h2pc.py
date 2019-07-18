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
import numbers
import pyAgrum.lib.ipython as gnb
import pyAgrum.lib.bn_vs_bn as comp
from sklearn import preprocessing
from pyAgrum.lib.bn2scores import computeScores
from independances import indepandance


class H2PC ():
    """H2PC is a new hybrid algorithm combining scoring and constraint-structured learning,
    which can be considered as an improvement of MMHC in many regards. Especially, it clearly enables
    to reduce the number of false negative edges.   
    
    
    """
   
    def __init__(self,learner,df,threshold_pvalue=0.05,verbosity=False,score_algorithm="greedy_climbing",optimized=False,filtering="AND",compute_number=False,**independance_args):
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
                raise ValueError ("we can't for the moment execute tests on databases with missing values")
            else: 
                #we convert each column as factor vectors
                le = preprocessing.LabelEncoder()
                self.df=df.apply(le.fit_transform,axis=0)
                
        
        
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
        
        if score_algorithm in ["greedy_climbing","tabu_search"]:
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
            self.tabu_size,self.nb_decrease=independance_args.get("tabu_size",100),independance_args.get("nb_decrease",20)
        
        self.learner=learner      
        if not isinstance(self.learner,gum.pyAgrum.BNLearner):
            raise TypeError("Only possible values for learner are pyAgrum.BNLearner or None") 
        
        self.independance_args=independance_args
        self.independance_args['learner']=learner       
        self.independance_args['levels']=self.df.nunique() 
        
        if not (isinstance(compute_number,bool)):
            raise TypeError("compute_test must a boolean value")
        else:
            self.compute_number=compute_number
      
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
            
            if self.verbosity:
                if ((variable_1 in neighbourhood_variable2) and (variable_2 not in neighbourhood_variable1)) or ((variable_1 not in neighbourhood_variable2) and (variable_2 in neighbourhood_variable1)):
                    print("there's an assymetry with variables {} and {} of respective neighbourds: {} and {} ".format(variable_1,variable_2, neighbourhood_variable1, neighbourhood_variable2))
             
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
        if self.score_algorithm=='greedy_climbing':
            self.learner.useGreedyHillClimbing()
        else:
            self.learner.useLocalSearchWithTabuList(self.tabu_size,self.nb_decrease)       
        bn_learned=self.learner.learnBN()  
        return bn_learned
    
    
    def _HPC_global(self):
        dico_couverture_markov={}
        for target in self.variables:  
            
            dico_couverture_markov[target]=hpc(target,self.df,self.threshold_pvalue,self.verbosity,whitelisted=self.whitelisted,blacklisted=self.blacklisted,**self.independance_args).couverture_markov()           
            
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
            dico_couverture_markov[target]=hpc(target,self.df,self.threshold_pvalue,self.verbosity,whitelisted=self.whitelisted,blacklisted=self.blacklisted,known_bad=known_bad,known_good=known_good,**self.independance_args).couverture_markov()["neighbours"]
            if self.verbosity:
                print("We compute with HPC the neighbours of '{}' :'{}' \n\n".format(target,dico_couverture_markov[target]) )
            
        return dico_couverture_markov
                
                    
                    
    
    
    def learnBN(self):
        #init number tests to 0
        indepandance.number_tests=0
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
        
        #score_based learning according to input_score
        self._add_set_unique_possible_edges(unique_possible_edges)   
        bn_learned=self._learn_graphical_structure()  
        if self.compute_number:
            ("we return both bn learnt and number of statistic tests performed")
            return (bn_learned,indepandance.number_tests)
        else:
            return bn_learned
  
   
        
    
        
        
        
        
        
        
if __name__ == "__main__":
    """
    asia_bn=gum.loadBN(os.path.join("true_graphes_structures","asia.bif"))
    #gnb.showBN(asia_bn)
    df=pd.read_csv(os.path.join("databases","sample_asia.csv"))
    learner=gum.BNLearner(os.path.join("databases","sample_asia.csv"))
    temps_debut=time.time()
    bn_H2PC=H2PC(learner,df,score_algorithm="tabu_search",optimized=False,filtering="AND",usePyAgrum=True)  
    #print("le temps d'exceution est {} .".format(time.time()-temps_debut))
    
    
   
    
    
    """
    alarm_bn=gum.loadBN(os.path.join("true_graphes_structures","alarm.bif"))
    #gnb.showBN(alarm_bn,"6")    
    df=pd.read_csv("sample_alarm.csv")
    computeScores(alarm_bn,"sample_alarm.csv")
    
    
    
    
    learner=gum.BNLearner("sample_alarm.csv")
    bn_H2PC_alarm=H2PC(learner,df,score_algorithm="tabu_search",optimized=False,filtering="AND",usePyAgrum=True).learnBN()    
    gnb.showBN(bn_H2PC_alarm)
    
    
    learner=gum.BNLearner("sample_alarm.csv")
    learner.useMIIC()
    bn_miic=learner.learnBN()
    gnb.showBN(bn_miic)
    
    print("comparaisosn entre miic et h2pc ", comp.GraphicalBNComparator(bn_miic,alarm_bn).scores())
    print("comparaisosn entre normal et h2pc ", comp.GraphicalBNComparator(bn_H2PC_alarm,alarm_bn).scores())
    

    
    
    
    



    
    
    
    
  
    
  

    
    
    
    

   
    
   
    
 

    

    
 
    
    
    
        
            
        
        
    
    
    