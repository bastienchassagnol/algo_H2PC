# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:01:32 2019

@author: bchassagno
"""

import os 
import pyAgrum as gum
import pyAgrum.lib._utils.oslike as oslike
import re
import time
import pyAgrum.lib.bn_vs_bn as comp
from h2pc import H2PC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_average_distance(number_repetitions,temp_database,size,distance_to_compute,true_bn):
    temp_scoring_matrix=np.empty((3,number_repetitions))
    print("we compute scoring for size ",size,"\n\n")
    for repetition in range(number_repetitions):    
        oslike.head(source_database,size,temp_database)
        
        #to get the size of each file computed
        #oslike.wc_l(os.path.join("out","extract_asia.csv"))
        learner=gum.BNLearner(temp_database) 
        
        #learn with greedy hill climbing
        learner.useGreedyHillClimbing()
        bn_greedy=learner.learnBN()
        temp_scoring_matrix[0,repetition]=comp.GraphicalBNComparator(bn_greedy,true_bn).scores()[distance_to_compute]
        
        #learn with tabu search
        learner.useLocalSearchWithTabuList()
        bn_tabu=learner.learnBN()
        temp_scoring_matrix[1,repetition]=comp.GraphicalBNComparator(bn_tabu,true_bn).scores()[distance_to_compute]
        
        #learn with H2PC
        bn_H2PC=H2PC(learner,score_algorithm="Greedy_climbing").learnBN()
        oslike.rm(temp_database)
        temp_scoring_matrix[2,repetition]=comp.GraphicalBNComparator(bn_H2PC,true_bn).scores()[distance_to_compute]
        
    print("valeur moyenne for size {} is {}".format(size,list(np.mean(temp_scoring_matrix,axis=1))))
    return list(np.mean(temp_scoring_matrix,axis=1))
    

def choose_graph_name(name_graphes):
    graph=input("choose one of the following graphs {} :\n".format(name_graphes))
    graph=re.sub(r'\W+', '', graph).lower()
    if graph in name_graphes:
        return graph
    else:
        choose_graph_name(name_graphes)

def generate_databases(structure_to_learn):
    #create for each size 30 databases and then store the results in a pandas database
    true_bn=gum.loadBN(os.path.join("true_graphes_structures",structure_to_learn+".bif"))
    
    source_database=os.path.join("databases", "sample_"+structure_to_learn+"_"+str(time.strftime("%A_%d_%B_%Y"))+ ".csv")
    print("name CSV file is ",source_database)
    gum.generateCSV(true_bn,source_database,500000)
    return source_database
            
def learn_scores(structure_to_learn,source_database,distance_to_compute='dist2opt'):    
    true_bn=gum.loadBN(os.path.join("true_graphes_structures",structure_to_learn+".bif"))
    scoring_values={algo:[] for algo in ['greedy_climbing','tabu_search','H2PC']}
     
    possible_scoring_distances=['recall','precision','fscore','dist2opt']
    if distance_to_compute not in possible_scoring_distances:
        raise AssertionError("distance score still not implemented, list of of possible computations is {}".format(possible_scoring_distances))
        
    for size in sample_size:
        temp_database=os.path.join("databases","extract_"+structure_to_learn+".csv")
        score_by_algorithm=compute_average_distance(30,temp_database,size,distance_to_compute,true_bn)        
        scoring_values['greedy_climbing'].append(score_by_algorithm[0])
        scoring_values['tabu_search'].append(score_by_algorithm[1])
        scoring_values['H2PC'].append(score_by_algorithm[2])
     #store results in a pandas object
    print("les valeurs de score sont les suivantes ",scoring_values)
    scoring_values_df=pd.DataFrame.from_dict(scoring_values)   
    return scoring_values_df
        
def plot_vizualisation(dataframe,sample_size,distance_to_compute):   
    fig, ax = plt.subplots(1, 1)
    pd.plotting.table(ax, np.round(score_results.iloc[:, [0,1,2]], 2),loc='lower right',colWidths=[0.15, 0.15, 0.15])
    dataframe.plot(ax=ax,x='sample_size', y=dataframe.columns.values[0:-1],style=['-', '--', '-.'],title="{} according to the size of databases".format(distance_to_compute))
       
    
  
        
        

#several sample sizes to look after
sample_size=[500,1000,2000,5000,20000]
#storing of true graph structures
name_graph_files=os.listdir("true_graphes_structures")
name_graphes=[os.path.splitext(graph)[0] for graph in name_graph_files]
structure_to_learn=choose_graph_name(name_graphes)

distance_to_compute='dist2opt'

#generate corresponding databases
source_database=generate_databases(structure_to_learn)
score_results=learn_scores(structure_to_learn,source_database,distance_to_compute)
score_results = score_results.assign(sample_size=pd.Series(sample_size).values) 

plot_vizualisation(score_results,sample_size,distance_to_compute)





    




