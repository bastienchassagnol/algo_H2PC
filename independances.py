# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:18:08 2019

@author: Bastien
"""



import pandas as pd
import pyAgrum as gum
from sklearn import preprocessing
import numbers
import os
import numpy as np
from scipy import stats 
import math
import matplotlib.pyplot as plt
import pyAgrum.lib.ipython as gnb
import random as rd

from utils import compute_time, format_test,format_args_indep



class indepandance ():
    """class attribute incrementing at each creation of the object"""
    number_tests = 0 # Le compteur vaut 0 au départ
    
    def __init__(self,df,ind_x,ind_y,conditions=[],reset_compteur=False,**independance_args):
        indepandance.number_tests += 1
        self.usePyAgrum=independance_args.get('usePyAgrum',False)
        if not isinstance(self.usePyAgrum,bool):
            raise TypeError("Only possible values for usePyAgrum are boolean")           
            
        self.dof_adjustment=independance_args.get('dof_adjustment',"classic")
        if self.dof_adjustment not in ["classic","adjusted","permut","permut_adjusted","sp"]:
            raise ValueError("Possible tests are included in these tests: {}".format(["classic","adjusted","permut","permut_adjusted","semi_parametric"]))
        
        self.calculation_method=independance_args.get('calculation_method',"pearson")
        if self.calculation_method not in ["pearson","log-likelihood","freeman-tukey","mod-log-likelihood","cressie-read"]:
            raise ValueError("Possible compuation statistics are included in these tests: {}".format(["pearson","log-likelihood","freeman-tukey","mod-log-likelihood","cressie-read"]))
            
            
        if self.usePyAgrum:
            self.learner=independance_args.get('learner')        
            #check if we use Pyagrum methods or not for independances   
            if isinstance(self.learner,gum.pyAgrum.BNLearner) and self.calculation_method not in ["pearson","log-likelihood"]:               
                raise ValueError("Only valuable tests are G2 and pearson with Pyagrum libraries")  
                
        self.verbosity=independance_args.get('verbosity',False)    
        if not isinstance (self.verbosity, bool):
            raise TypeError("Expected format for verbosity is boolean")      
       
        
        self.power_rule=independance_args.get('power_rule')
        if self.power_rule:
            if not isinstance (self.power_rule,numbers.Number):
                raise TypeError("Format expected for power rule is a number.")          
        
                
        self.threshold_pvalue=independance_args.get('threshold_pvalue',0.05)        
        if isinstance (self.threshold_pvalue,numbers.Number):            
            if (self.threshold_pvalue>1) or (self.threshold_pvalue<0):
                raise ValueError("Probability of threshold value must be included in range [0,1]")            
        else:
            raise TypeError ("Format expected for threshold pavlue must be a number")
        
        #implement number of permutations and threshold values
        if "permut" in self.dof_adjustment:
            self.permutation_number=independance_args.get("permutation_number",5000)
        if "adjusted" in self.dof_adjustment:
            self.normal_confidence=independance_args.get("normal_confidence",0.95)
            if (self.normal_confidence>1) or (self.normal_confidence<0):
                raise ValueError("Probability of normal confidence must be included in range [0,1]")   
                             
        if "sp" in self.dof_adjustment:
            self.permutation_number=independance_args.get("permutation_number",150)    
        
        self.R_test=independance_args.get('R_test',False)  
        if isinstance(self.R_test,bool):
            if self.calculation_method not in ["pearson","log-likelihood"]:
                raise ValueError("Only possible tests computed with R tests are pearson and log likelihood tests")
        else:
            raise TypeError("Expected format for R_test is boolean")   
        
        
        liste_variables=list(df.columns) 
        #check nature of variables for independance puroposes, possible values are integer indexes or variable names of df
        
        self.ind_x=self.check_value(ind_x,liste_variables)            
        self.ind_y=self.check_value(ind_y,liste_variables)         
        self.ind_z=list({self.check_value(indice,liste_variables) for indice in conditions})
        
        
        #check that variables to check are not in the condtionning set
        if self.ind_x in self.ind_z or self.ind_y  in self.ind_z:            
            raise ValueError("duplication of either x or y in the condition set") 
        
        #controle empty values and nature of dataframe
        #then, we only select columns of interest
        
        if not isinstance(df,pd.core.frame.DataFrame):
           raise TypeError ("expected format is a dataframe")
        else:
            self.df=df   
        
        
        
        if not self.usePyAgrum and not self.R_test:
            self.levels=independance_args.get('levels',self.df.nunique())
            
        self.bn=independance_args.get('bn') 
        if self.bn:
            if not isinstance (self.bn,gum.pyAgrum.BayesNet):      
                raise ValueError("Bayesian network must be of that type : pyAgrum.pyAgrum.BayesNet or None")  
      
    
    def check_value(self,indice,liste_variables):
        if isinstance(indice,int):
            if  indice<len(self.df.columns) and indice>=0:
                    return liste_variables[indice]
            else:
                raise ValueError("index is supposed included between 0 and number of variables")
        elif isinstance(indice,str):
            if indice in liste_variables:
                return indice                
            else:
                raise ValueError ("You try to condition on a not existing value: {}".format(indice))            
        else:
            raise TypeError("Format expected for condition test is either a string or the integer index of the variable")
    
   
        
      
        
    def apply_heuristic_one(self):        
        return len(self.df.index) < (self.power_rule * self.levels_x * self.levels_y * self.levels_z)   
        
    def compute_statistic(self,group,adjust_degree=False,total_df=None,permut=False): 
        
        if permut:
            group=group.copy()
            group[self.ind_y]=np.random.permutation(group[self.ind_y].values)    
        effectif_observe_by_z=np.array(pd.crosstab(group[self.ind_x],group[self.ind_y])) 
        chi2_stat_by_z =stats.chi2_contingency(effectif_observe_by_z, correction=False, lambda_=self.calculation_method)[0]
                
        #computing in the same time degrees of freedom adjustement
        if adjust_degree:
            #adjust with null XZ frequencies
            total_df-=((self.levels_x-effectif_observe_by_z.shape[0])*(self.levels_y-1))
            #adjust with null YZ frequencies
            total_df-=((self.levels_y-effectif_observe_by_z.shape[1])*(self.levels_x-1))
            #adjust with nul XYZ frequencies, whom XZ or YZ are not null
            #in our case, equivalent tou count the number of zeros in the matrix
                        
            return chi2_stat_by_z,total_df   
        return chi2_stat_by_z  
    
    def size_group(self,group):
        return (len(group))
        
    
    def classic_test(self):      
        #we sum statistic for each group Z 
        statistic=0
        if self.ind_z:
            grouped=self.df.groupby(self.ind_z, as_index=False)
            for name, group in grouped:
                 statistic+=self.compute_statistic(group)
        else:
            statistic=self.compute_statistic(self.df)
        dof=(self.levels_x-1)*(self.levels_y-1)*self.levels_z
        p_value=1-stats.chi2.cdf(statistic,dof)        
        return (statistic,p_value,dof)  
        
    def adjusted_test(self,condition_df): 
        if self.ind_z:
            groups,adjusted_df,chi2=condition_df.groupby([self.ind_z]),(self.levels_x-1)*(self.levels_y-1)*self.levels_z,0       
            for  name,group in groups:            
                chi_temp,adjusted_df=self.compute_statistic(group,adjust_degree=True,total_df=adjusted_df)
                chi2+=chi_temp
        else:
            adjusted_df=(self.levels_x-1)*(self.levels_y-1)*self.levels_z
            chi2=self.compute_statistic(group,adjust_degree=True,total_df=adjusted_df)
        p_value=1-stats.chi2.cdf(chi2,adjusted_df)
        return (chi2,p_value,adjusted_df)
         
        
    def permutation(self,condition_df,semi_parametric_option=False):        
        #first compute chi2 statistic in the original database
        chi2_original=self.classic_test()[0]
        #store stats for each permuation
        chi2_permut_vector=np.empty(self.permutation_number)
        #conditional permutation test
        
        for permutation in range (self.permutation_number):    
            if self.ind_z:                                      
                chi2_permut=0          
                for  name,group in condition_df.groupby([self.ind_z]):  
                   chi2_permut+=self.compute_statistic(group,permut=True)
            else:
                chi2_permut=self.compute_statistic(group,permut=True)
            chi2_permut_vector[permutation]=chi2_permut                
               
        if semi_parametric_option:
            df_adjusted=np.mean(chi2_permut_vector)            
            return (chi2_original, 1-stats.chi2.cdf(chi2_original,df_adjusted,df_adjusted))
        else:
            return (chi2_original,np.count_nonzero(chi2_original<chi2_permut_vector)/self.permutation_number,None)
        
    def semi_parametric(self):
        return self.permutation(semi_parametric_option=True)
    
    def heuristic_permutation(self,condition_df):
        chi2_0, dof_0, p_0=self.classic_test()     
        #apply heuristic one 
        if (p_0 >0.5 or p_0<0.001):
            return (chi2_0, dof_0, p_0)
        #store stats for each permutation
        chi2_permut_vector=np.array([])
        
        for permutation in range (self.permutation_number):  
            if self.ind_z:
                chi2_permut=0            
                for  name,group in condition_df.groupby([self.ind_z]):                     
                   chi2_permut+=self.compute_statistic(group,permut=True)   
            else:
                chi2_permut=self.compute_statistic(group,permut=True)  
            chi2_permut_vector=np.append(chi2_permut_vector,chi2_permut)
            #apply heuristic 2
            estimated_p_value=np.count_nonzero(chi2_0<chi2_permut_vector)/(permutation+1)
            maximum_magnitude=(stats.norm.ppf((1+self.normal_confidence)/2))/(2*math.sqrt(permutation+1))            
            if (estimated_p_value+maximum_magnitude)<self.threshold_pvalue:
                return (chi2_0, estimated_p_value,None)
            elif ((estimated_p_value-maximum_magnitude)>self.threshold_pvalue) :
                return (chi2_0,  estimated_p_value,None)
            
            #apply heuristic 3
            upper_bound,lower_bound=1-np.count_nonzero(chi2_0>chi2_permut_vector)/self.permutation_number,np.count_nonzero(chi2_0<chi2_permut_vector)/self.permutation_number
            if upper_bound<self.threshold_pvalue:                
                return (chi2_0,  estimated_p_value,None)
            elif lower_bound>self.threshold_pvalue:                
                return (chi2_0,  estimated_p_value,None)   

        return (chi2_0,np.count_nonzero(chi2_0<chi2_permut_vector)/self.permutation_number,None)
    
    
    
    def realize_R_test(self,type_test):
      #condition_df=self.column_treatment()           
      #several useful imports to make type transition between Python and R
       
      #pandas2ri uses a library required to convert Python dataframe to R dataframes
      from rpy2.robjects import r, pandas2ri,StrVector
      pandas2ri.activate()
      r_dataframe=pandas2ri.py2ri(self.df)
      converted_ind_z=StrVector(self.ind_z)
      
      r('''
        # execute ci.test
        require (bnlearn)
        f <- function(x,y ,z=c(), df,test) {
        
        #convert dataframe as factor df
        for(i in 1:ncol(df)){
        df[,i] <- as.factor(df[,i])
        }
        #two types of tests, according whether z is present or not    
        if (is.null(z)) { resultat=ci.test(x=x,y=y,data=df,test=test) }
        else {resultat=ci.test(x=x,y=y,z=z,data=df,test=test) }                
        return (c(as.numeric(resultat$statistic),as.numeric(resultat$p.value),as.numeric(resultat$parameter[1])))                
        }
        ''')
      ci_test = r['f']
      #condtionnal test
      if self.ind_z:
          return ci_test(x=self.ind_x,y=self.ind_y,z=converted_ind_z,df=r_dataframe, test=type_test)
      #classic independance test
      else:          
          return ci_test(x=self.ind_x,y=self.ind_y,df=r_dataframe, test=type_test)
        
        
    
    
    def realize_test(self): 
        if self.verbosity:
           print("Statistic test carried out is {} with degrees adjustement {} and following threshold value {}".format(self.calculation_method, self.dof_adjustment, self.threshold_pvalue))
        if self.usePyAgrum: 
            type_test={"pearson":self.learner.chi2,"log-likelihood":self.learner.G2}           
            retour=type_test[self.calculation_method](self.ind_x,self.ind_y,self.ind_z)
            if self.verbosity:
                print("Computed values are, in that order, stat computed: {}, and pvalue: {}.".format(*retour))
            
        elif self.R_test:
            #dico to convert Python parameters to R parameters
            python_test=self.dof_adjustment+"_"+self.calculation_method
            dico_conversion_type={"classic_pearson": "x2","adjusted_pearson":"x2-adf","permut_pearson":"mc-x2","permut_pearson":"mc-x2","sp_pearson":"sp-x2",\
                                  "classic_log-likelihood": "mi","adjusted_log-likelihood":"mi-adf","permut_log-likelihood":"mc-mi","permut_log-likelihood":"mc-mi","sp_log-likelihood":"sp-mi"}
            
            #two several types of tests, according it's conditionnal independance or not
            
            
            retour=self.realize_R_test(dico_conversion_type[python_test])
            if self.verbosity:
                print("Computed values are, in that order, stat computed: {}, and pvalue: {}.".format(*retour))              
            
        else:
            
            type_test={ "classic":self.classic_test,"adjusted":self.adjusted_test,"permut":self.permutation,"permut_adjusted":self.heuristic_permutation,"sp":self.semi_parametric}
            #condition_df=self.column_treatment()            
            #apply heuristic one of power rule     
            if self.ind_z:
                self.levels_x,self.levels_y, self.levels_z=self.levels[self.ind_x],self.levels[self.ind_y],self.levels[self.ind_z].prod()                
            else:
                self.levels_x,self.levels_y, self.levels_z=self.levels[self.ind_x],self.levels[self.ind_y],1
            
            if self.power_rule and self.apply_heuristic_one():
                retour=math.inf,1,None            
            retour=type_test[self.dof_adjustment]()            
            if self.verbosity:
                print("Computed values are, in that order, stat computed: {}, pvalue: {} and degrees of freedom: {}".format(*retour))
        return retour 
    
    def get_all_combinations(self,list):
        #initialize with the levels of first index
        ancient_combinations=[[level] for level in range(list[0])]
        for i in range (1,len(list)):           
            nlevels_z=list[i] 
            new_combinations=[]
            for level in range (nlevels_z): 
                for combination in ancient_combinations:
                    temp=combination.copy()
                    temp.append(level)
                    new_combinations.append(temp)
            ancient_combinations=new_combinations.copy()
        return [''.join(str(e) for e in combination) for combination in new_combinations]
    
    def compute_exact_statistic(self):
        if self.bn:
            if not hasattr(self, 'attr_name'):
                self.levels=self.df.nunique()
            self.levels_x,self.levels_y, self.levels_z=self.levels[self.ind_x],self.levels[self.ind_y],self.levels[self.ind_z].sum()
            ie=gum.LazyPropagation(self.bn)
            #2 cases: either 2 variables to compute, or conditionnaly to z
            stat,nb_samples=0,self.df.shape[0]
            if self.ind_z:
                matrix_proba=ie.evidenceJointImpact([self.ind_x,self.ind_y],self.ind_z).toarray()
                valuable_z=list(matrix_proba[:-2].shape)
                all_combinations_z=self.get_all_combinations(valuable_z)
                #convert under the form of slices indexes
                all_indices_z=[tuple(slice(int(x),None,None) for x in combination) for combination in all_combinations_z]
                for slice_z in all_indices_z:
                    #we keep matrix probability for each level of z, multiplied by the number of rows to convert probabilities into real sizes
                    matrix_z=matrix_proba[slice_z]
                    stat+=stats.chi2_contingency(matrix_z, correction=False, lambda_=self.calculation_method)[0]*nb_samples
            else:
                #in case of only 2 variables
                ie.addJointTarget({self.ind_x,self.ind_y})
                matrix_proba=ie.jointPosterior([self.ind_x,self.ind_y]).toarray()
                stat=stats.chi2_contingency(matrix_proba, correction=False, lambda_=self.calculation_method)[0]*nb_samples                 
            # we then return the exact theoretical pvalue expected  
            dof=(self.levels_x-1)*(self.levels_y-1)
            return 1-stats.chi2.cdf(stat,dof) 
        else:
            raise TypeError("we can't compute exact theoretical test without bayesian network")
    
    
    
    

    
    
        
        
        
def compute_independance_tests(bn,sizes,test,nb_test=20,**dico_independance):
    """
    Using $nb_test$ database of size $size$ from the bn $bn$, 
    computing the p-value for a list $lindep$ of conditional independence tests, using $test$ type.
    """
    #pvalue vector stores pvalues for a given size for several conditions
    pvalue_vector=[]    
    #pvalue amplitude stores the shift between max and min pvalue measured
    pvalue_min_error,p_value_max_error=[],[]
    le = preprocessing.LabelEncoder()
    
    usePyAgrum=dico_independance.get('usePyAgrum',False)
    for size in sizes:    
        pvalue_temp=np.empty(nb_test)
        for indice in range(nb_test):
            gum.generateCSV(bn,os.path.join("databases","sample_score.csv"),size,False)
            df=pd.read_csv(os.path.join("databases","sample_score.csv"))
            df=df.apply(le.fit_transform,axis=0)
            #we have to fit learner with the new randomly created df
            if usePyAgrum:
                dico_independance["learner"]=gum.BNLearner(os.path.join("databases","sample_score.csv"))
            pvalue_temp[indice]=indepandance(df,*test,**dico_independance).realize_test()[1]
        
        pvalue_vector.append(np.mean(pvalue_temp))
        pvalue_min_error.append((np.mean(pvalue_temp)-np.min(pvalue_temp)))
        p_value_max_error.append((np.max(pvalue_temp)-np.mean(pvalue_temp)))
    return pvalue_vector,np.array([pvalue_min_error,p_value_max_error])

def plot_independance_tests(df,bn,sizes,lindep,nb_test=20,list_args_independance=[]):  
    """
    Subplot p-value for a list $lindep$ of conditional independence tests, using independance criterion 
    $list_args_independance$ for each subplot.
    """
    fig,ax = plt.subplots(nrows=len(list_args_independance), ncols=1, sharex=True,figsize =[6.4, 2*len(list_args_independance)])
    #define set of colors
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0.2, 1, len(lindep))]
    for index, indep_criterion in enumerate(list_args_independance):
        for color,test in zip(colors,lindep):
            #compute exact expected statistic value and modifying a bit true value to display all lines
            true_value=indepandance(df,*test,**indep_criterion,bn=bn).compute_exact_statistic()+(rd.random()*0.1-0.05)        
            pvalues,pvalues_err=compute_independance_tests(bn,sizes,test,nb_test,**indep_criterion)            
            ax[index].errorbar(sizes, pvalues, yerr=pvalues_err, uplims=True, lolims=True, label=format_test(test),color=color)
            #ax[index].axhline(true_value,linewidth=0.5,color=color)
            ax[index].set_ylabel ("pValue")
            ax[index].set_title(format_args_indep(indep_criterion))
    
    

    #at the end of computation, delete temporay file created
    if os.path.exists(os.path.join("databases","sample_score.csv")):
        os.remove(os.path.join("databases","sample_score.csv"))    
    plt.tick_params(axis='x',rotation=45)
    plt.xlabel ("data size") 
    box = ax[len(list_args_independance)-1].get_position()
    ax[len(list_args_independance)-1].set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])    
    ax[len(list_args_independance)-1].legend(bbox_to_anchor=(0, -1,1.5, .102), loc=3,ncol=2, mode="expand", fancybox=True, shadow=True)    
    fig.suptitle("Independance tests carried out for several sizes",y=1.05,weight ="bold")
    fig.tight_layout()
    return fig

def plot_indep_time_computation(bn,lindep, ntimes,sizes,list_args_independance=[{"calculation_method":"pearson","R_test":True},{"calculation_method":"pearson"},{"usePyAgrum":True,"calculation_method":"pearson"}]):
    """
    Subplot time computation for a list $lindep$ of conditional independence tests carried on the same database $ntimes$
     for databases of sizes $sizes$, this done for each indep_criterion $list_args_independance$ given. 
    """
    
    fig,ax = plt.subplots(nrows=len(lindep), ncols=1, sharex=True,figsize =[6.4, 2*len(lindep)])
    
 
    #define set of colors
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0.2, 1, len(list_args_independance))]
    for index, indep_relation_ship in enumerate(lindep):        
        for color,test in zip(colors,list_args_independance):
            store_time,use_pyagrum=[],test.get("usePyAgrum",False)
            for size in sizes:
                #we generate df temporaly with the wanted size
                gum.generateCSV(bn,os.path.join("databases","temp_size.csv"),size,False,with_labels=True)
                df=pd.read_csv(os.path.join("databases","temp_size.csv"))
                #we generate a function with the expected return
                if use_pyagrum:
                    test["learner"]=gum.BNLearner(os.path.join("databases","temp_size.csv"))
                modified_function=compute_time(indepandance(df,*indep_relation_ship,**test).realize_test)
                temps_execution=0
                for iteration in range (ntimes):
                    temps_execution+=modified_function()
                store_time.append(temps_execution)
            
            ax[index].plot(sizes, store_time, label=format_args_indep(test),color=color,marker='+')
        ax[index].set_ylabel ("Time execution")
        ax[index].set_yscale('log')
        ax[index].set_title(format_test(indep_relation_ship))        
    
    #at the end of computation, delete temporay file created
    if os.path.exists(os.path.join("databases","temp_size.csv")):
        os.remove(os.path.join("databases","temp_size.csv")) 
    plt.tick_params(axis='x',rotation=45)
    plt.xlabel ("data size") 
    box = ax[len(lindep)-1].get_position()
    ax[len(lindep)-1].set_position([box.x0, box.y0 + box.height * 0.4,
                     box.width, box.height * 0.6])    
    ax[len(lindep)-1].legend(bbox_to_anchor=(0.0, -1,1., .102), loc=3, mode="expand", fancybox=True, shadow=True)    
    fig.suptitle("Times to carry out independance tests for several sizes of datasets, each computed {} times".format(ntimes),y=1.0,weight ="bold")
    fig.tight_layout()
    return fig
    







    




   
if __name__ == "__main__":   
    asia_bn=gum.loadBN(os.path.join("true_graphes_structures","asia.bif"))    
    learner=gum.BNLearner("sample_asia.csv")
    indep_criterion=[{"learner":learner, "usePyAgrum":True},{"learner":learner,"calculation_method":"log-likelihood","usePyAgrum":True}]
    df=pd.read_csv("sample_asia.csv")
    #♣plot_independance_tests(df,asia_bn,sizes,lindep,nb_test=20,list_args_independance=indep_criterion)
    
    lindep=[("asia","smoke",['lung']),    ("asia","smoke",[]),
                                          ("dysp","smoke",[]),
                                          ("dysp","smoke",["lung","bronc"]),
                                          ("tub","bronc",[]),
                                          ("tub","bronc",["dysp"])]
    
    sizes,ntimes=[200,1000,5000],10
    plot_indep_time_computation(asia_bn,lindep, ntimes,sizes,list_args_independance=[{"calculation_method":"log-likelihood"},{"usePyAgrum":True,"calculation_method":"pearson"}])

   
  
  
   
    
    
    

    
    
  


    

    
    
    
    
    
    
    
    
    
    
   
    
    