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
from utils import load_objects,save_objects
import pyAgrum.lib.ipython as gnb
from pyAgrum.lib.bn2scores import computeScores
import pickle
from independances import indepandance
from collections import OrderedDict

from matplotlib.lines import Line2D



def choose_graph_name(name_graphes):
    dico_name_graphes_formatted={os.path.splitext(graph)[0]:graph for graph in name_graphes}    
    graph=input("choose one of the following graphs {} :\n".format(list(dico_name_graphes_formatted.keys())))
        
    if graph in list(dico_name_graphes_formatted.keys()):
        return dico_name_graphes_formatted[graph]
    else:
        print("Inserted name does not belong to those written")
        choose_graph_name(name_graphes)
        
        
        


def compute_average_distance(bn, nsamples,size,algorithm,score_measured):
    #matrix of shape (repetitions * scores) to get the mean of each type of score
    #algorithm:(type,agrs,kwargs)
    type_algorithm,args,kwargs=algorithm[0],algorithm[1][0],algorithm[1][1]    
    scoring_matrix=np.empty(shape=(len(score_measured),nsamples))
    
    for repetition in range(nsamples):    
        #generate a database of the required size
        gum.generateCSV(bn,os.path.join("databases","temp_base.csv"),size,False,with_labels=True)  
        learner=gum.BNLearner(os.path.join("databases","temp_base.csv")) 
        dico_algorithm={'greedy_climbing':learner.useGreedyHillClimbing,'tabu_search':learner.useLocalSearchWithTabuList, 'miic':learner.useMIIC, '3off2':learner.use3off2}
        
               
        #with algos coming from pyagrum
        if type_algorithm in dico_algorithm.keys():
            detect_cycle,iteration=True,0
            #we add that part to avoid cycle erros by repeating the process
            while (detect_cycle and iteration<100):                
                start_time=time.time()
                iteration+=1    
                try:
                    learner.useMIIC()
                    created_bn=learner.learnBN()
                except gum.InvalidDirectedCycle:                    
                    gum.generateCSV(bn,os.path.join("databases","temp_base.csv"),size,False,with_labels=True) 
                    learner=gum.BNLearner(os.path.join("databases","temp_base.csv")) 
                    dico_algorithm[type_algorithm](*args,**kwargs)
                else:
                    detect_cycle=False
                    end_time=time.time()-start_time 
            if iteration>=100:
                raise AssertionError("failure of miic to compute the bayesian network")
                    
        #for the moment, only possible case is H2PC case, these lines won't be required after
        else:
            df=pd.read_csv(os.path.join("databases","temp_base.csv"))
            start_time=time.time()
            created_bn=H2PC(learner,df,*args,**kwargs).learnBN()
            end_time=time.time()-start_time
            
        #store results from the scores pyagrum tools in scoring_matrix
        
        scores_list=comp.GraphicalBNComparator(bn,created_bn).scores()
        scores_list.update(computeScores(created_bn,os.path.join("databases","temp_base.csv")))
        scores_list.update({'time':end_time,'number_tests':indepandance.number_tests,'specificity':scores_list['count']['tn']/(scores_list['count']['tn'] +scores_list['count']['fp'])})
        #gather all scores in a single list
        
        #store results of each score, conserving the order of the given list
        ordered_score = OrderedDict((score, scores_list[score]) for score in score_measured)
        scoring_matrix[:,repetition]=list(ordered_score.values())
        
    return scoring_matrix
    
        
        
        
def learn_scores(bn,sample_size,score_measured=['dist2opt'],algorithms={'tabu_search':([],{})},nsamples=30):    
    possible_scoring_distances=['recall','precision','fscore','dist2opt','bic','aic','mdl','time','number_tests','specificity']
    for score in score_measured:
        if score not in possible_scoring_distances:
            raise AssertionError("distance score still not implemented, list of of possible computations is {}".format(possible_scoring_distances))
    #assert that if there's number_tests, there's only h2pc algo used
    if ('number_tests' in score_measured) and "".join(list(algorithms.keys()))!='h2pc':
        raise AssertionError ("we can only compute number of tests for h2pc")
    possible_algorithms=['greedy_climbing','tabu_search', 'H2PC', 'miic', '3off2']
    for algo in algorithms: 
        if algo not in possible_algorithms:
            raise AssertionError("algorithm score still not implemented, list of of possible computations is {}".format(list(possible_algorithms.keys())))
    
    #matrix scoring all scores measured for each database
    matrix_scores=np.empty(shape=(len(sample_size),len(algorithms),len(score_measured),nsamples))
    for row_index, size in enumerate (sample_size):
        for column_index, algorithm in enumerate(algorithms.items()):
            print("nous en sommes a l'algo ", algorithm, "pour al taille suivante de database ", size)
            matrix_scores[row_index,column_index,Ellipsis]=compute_average_distance(bn, nsamples,size,algorithm,score_measured)
    return matrix_scores

   


def plot_score_algorithms(bn_name,sample_size,score_measured=['dist2opt'],algorithms={'tabu_search':([],{})},nsamples=10,with_boxplot=False):  
    """
    Subplot score algorithms from a matrix score ( size*algo*scores).
    """
   
    bn=gum.loadBN(os.path.join("true_graphes_structures",bn_name))
    matrix_score=learn_scores(bn,sample_size,score_measured,algorithms,nsamples)
    
    
    save_objects(os.path.join('scores','matrice_scores_{}'.format(os.path.splitext(bn_name)[0])),matrix_score)
   
    
    #matrix_score=load_objects(os.path.join('scores','matrice_scores_alarm.bif'))
    
    fig,ax = plt.subplots(nrows=len(score_measured), ncols=1, sharex=True,figsize =[6.4, 2*len(score_measured)])
    #define set of colors
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0.1, 1, len(score_measured))]
    mean_colors=colors[::-1]    
    for index_score, score in enumerate(score_measured):
        for index_algo,mixed in enumerate(zip(colors,algorithms)):            
            score_repetition=matrix_score[:,index_algo,index_score]
            mean_resampling=np.mean(score_repetition,axis=1)
            if with_boxplot:
                meanpointprops = dict(marker='D', markeredgecolor='white',markerfacecolor=mean_colors[index_score])
                bp=ax[index_score].boxplot(np.transpose(score_repetition),notch=True, sym=' ',patch_artist=True,showmeans=True,meanprops=meanpointprops)                
                for box in bp['boxes']:
                    # change fill color
                    box.set( facecolor = colors[index_score] )
            else:
                ax[index_score].plot(sample_size,mean_resampling, label=mixed[1],color=mixed[0],marker='+')
        ax[index_score].set_ylabel (score)
        ax[index_score].set_title("Score measure {} in relation with dataset size".format(score))
        #at the end of computation, delete temporay file created
        
    if os.path.exists(os.path.join("databases","temp_base.csv")):
        os.remove(os.path.join("databases","temp_base.csv"))  
    plt.tick_params(axis='x',rotation=45)
    plt.xlabel ("data size") 
    box = ax[len(score_measured)-1].get_position()
    ax[len(score_measured)-1].set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])    
    ax[len(score_measured)-1].legend(bbox_to_anchor=(0.0, -1,1., .102), loc=3,ncol=4, mode="expand", fancybox=True, shadow=True)    
    fig.suptitle('Some score measures for BN : {}'.format(os.path.splitext(bn_name)[0]),y=1.05,weight ="bold")
    fig.tight_layout()
    return fig
    
    
    





def compute_ratio(bn, nsamples,size,algorithms,score_measured):    
    matrix_ratio_list=[compute_average_distance(bn, nsamples,size,algo,score_measured) for algo in algorithms.items()]      
    #we add a really small value to avoid divisions by 0
    matrix_ratio=matrix_ratio_list[1]/(matrix_ratio_list[0]+10**-10)
    return matrix_ratio
 
    
def learn_ratio(bn,sample_size,score_measured=['dist2opt'],algorithms={'tabu_search':([20,50],{}),'greedy_climbing':([],{})},nsamples=30):    
    possible_ratio_distances=['recall','precision','fscore','dist2opt','bic','aic','mdl','time','number_tests','specificity']
    for score in score_measured:
        if score not in possible_ratio_distances:
            raise AssertionError("distance score still not implemented, list of of possible computations is {}".format(possible_ratio_distances))
      
    if len (algorithms)!=2:
        raise AssertionError("ratio is supposed to be between 2 distances only")
    possible_algorithms=['greedy_climbing','tabu_search', 'H2PC', 'miic', '3off2']
    for algo in algorithms: 
        if algo not in possible_algorithms:
            raise AssertionError("algorithm score still not implemented, list of of possible computations is {}".format(list(possible_algorithms.keys())))
    
    #matrix scoring all scores measured for each database
    matrix_scores=np.empty(shape=(len(sample_size),len(score_measured),nsamples))
    for index_size, size in enumerate (sample_size):
        print("nous en sommes a la taille ",size)
        matrix_scores[index_size,Ellipsis]=compute_ratio(bn, nsamples,size,algorithms,score_measured)
    return matrix_scores

def plot_ratio_algorithms(bn_name,sample_size,score_measured=['dist2opt'],algorithms={'tabu_search':([],{})},nsamples=30):  
    """
    Subplot score algorithms from a matrix score ( size*algo*scores).    """   
    print("nom bn est ", bn_name)
    bn=gum.loadBN(os.path.join("true_graphes_structures",bn_name))
    matrix_ratio=learn_ratio(bn,sample_size,score_measured,algorithms,nsamples)
    
    """
    save_objects('matrice_scores',matrix_score)
    matrix_score=load_objects('matrice_scores')
    """
     
    fig,ax = plt.subplots(nrows=len(score_measured), ncols=1, sharex=True,figsize =[6.4, 2*len(score_measured)])
    #define set of colors
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0.1, 1, len(score_measured))]
    mean_colors=colors[::-1]
    for index_score, score in enumerate(score_measured):
        score_repetition=matrix_ratio[:,index_score,:]
        meanpointprops = dict(marker='D', markeredgecolor='white',markerfacecolor=mean_colors[index_score])
        bp=ax[index_score].boxplot(np.transpose(score_repetition),notch=True, sym=' ',patch_artist=True,showmeans=True,meanprops=meanpointprops)
        for box in bp['boxes']:
            box.set( facecolor = colors[index_score] )        
        ax[index_score].set_ylabel (score)
        ax[index_score].set_title("Score ratio measure {} in relation with dataset size".format(score))
        ax[index_score].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
        
        
        
    ax[len(score_measured)-1].set_xticklabels(sample_size,rotation=45, fontsize=8)
        
    #at the end of computation, delete temporay file created    
    if os.path.exists(os.path.join("databases","temp_base.csv")):
        os.remove(os.path.join("databases","temp_base.csv"))  
    
    box = ax[len(score_measured)-1].get_position()
    ax[len(score_measured)-1].set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])   
    
    legend_elements=[Line2D([0], [0], marker='D', markeredgecolor='white',markerfacecolor=mean_colors[index_score],color="w",label="mean ratio for score: {}".format(score_measured[index_score])) for index_score in range (len(score_measured))]    
    
    ax[len(score_measured)-1].legend(handles=legend_elements,bbox_to_anchor=(0.0, -1,1., .102), loc=3,ncol=4, mode="expand", fancybox=True, shadow=True)    
    plt.xlabel("data_size")
    fig.suptitle('Ratio of scores of {} over {} for BN : {}'.format(list(algorithms.keys())[1],list(algorithms.keys())[0],os.path.splitext(bn_name)[0]),y=1.05,weight ="bold")
    fig.tight_layout()
    return fig


  
        
        
if __name__ == "__main__":
    #several sample sizes to look after
    sample_size=[200,500,1000]
    #storing of true graph structures
    name_graph_files=os.listdir("true_graphes_structures")
    
    bn=choose_graph_name(name_graph_files)
    #print("le vrai bn est ",bn)
    
    
    
    #fig=plot_score_algorithms(bn,sample_size,['recall','fscore','dist2opt','specificity'],algorithms={'miic':([],{}),'H2PC':([],{'optimized':False,'filtering':"AND",'usePyAgrum':True})},nsamples=30)
    #plt.savefig(os.path.join("figures","asia_scores"))
    
    #fig=plot_ratio_algorithms(bn,sample_size,score_measured=['dist2opt','recall'],algorithms={'tabu_search':([20,50],{}),'H2PC':([],{'optimized':False,'filtering':"AND",'usePyAgrum':True})},nsamples=10)
    #plt.savefig(os.path.join("figures","asia_ratios"))
    
   
    
    


    
                   
    
    










