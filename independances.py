# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:18:08 2019

@author: Bastien
"""
import pandas as pd
import pyAgrum as gum
from sklearn import preprocessing
import numpy as np
from scipy import stats 
import numbers
import math
import os
import pyAgrum.lib.ipython as gnb
from functools import partial
import matplotlib.pyplot as plt




class indepandance ():
    
    def __init__(self,df,ind_x,ind_y,conditions=[],**independance_args):
        
        
        #controle empty values and nature of dataframe
        if not isinstance(df,pd.core.frame.DataFrame):
           raise TypeError ("expected format is a dataframe")
        else:
            if df.isnull().values.any():
                raise ValueError ("we can't perform tests on databases with missing values")
            else:
                self.df=df                
        liste_variables=list(df.columns)
        
        
        self.dof_adjustment=independance_args.get('dof_adjustment',"classic")
        if self.dof_adjustment not in ["classic","adjusted","permut","permut_adjusted","sp"]:
            raise ValueError("Possible tests are included in these tests: {}".format(["classic","adjusted","permut","permut_adjusted","semi_parametric"]))
        
        self.calculation_method=independance_args.get('calculation_method',"pearson")
        if self.calculation_method not in ["pearson","log-likelihood","freeman-tukey","mod-log-likelihood","cressie-read"]:
            raise ValueError("Possible compuation statistics are included in these tests: {}".format(["pearson","log-likelihood","freeman-tukey","mod-log-likelihood","cressie-read"]))
                   
        
        self.learner=independance_args.get('learner')        
        #check if we use Pyagrum methods or not for independances   
        if not isinstance(self.learner,gum.pyAgrum.BNLearner) and self.learner is not None:
            raise TypeError("Only possible values for learner are pyAgrum.BNLearner or None") 
        
        if isinstance(self.learner,gum.pyAgrum.BNLearner) and self.calculation_method not in ["pearson","log-likelihood"]:               
            raise ValueError("Only valuable tests are G2 and pearson with Pyagrum libraries")  
        
        
        #check nature of variables for independance puroposes, possible values are integer indexes or variable names of df
        self.ind_x=self.check_value(ind_x,liste_variables)            
        self.ind_y=self.check_value(ind_y,liste_variables)         
        self.ind_z=list({self.check_value(indice,liste_variables) for indice in conditions})
        
        
        
        #check that variables to check are not in the condtionning set
        if self.ind_x in self.ind_z or self.ind_y  in self.ind_z:
            
            raise ValueError("duplication of either x or y in the condition set") 
            
        self.verbosity=independance_args.get('verbosity',False)    
        if not isinstance (self.verbosity, bool):
            raise TypeError("Expected format for verbosity is boolean")
        
        self.dof_adjustment=independance_args.get('dof_adjustment',"classic")
        if self.dof_adjustment not in ["classic","adjusted","permut","permut_adjusted","sp"]:
            raise ValueError("Possible tests are included in these tests: {}".format(["classic","adjusted","permut","permut_adjusted","semi_parametric"]))
        
        self.calculation_method=independance_args.get('calculation_method',"pearson")
        if self.calculation_method not in ["pearson","log-likelihood","freeman-tukey","mod-log-likelihood","cressie-read"]:
            raise ValueError("Possible compuation statistics are included in these tests: {}".format(["pearson","log-likelihood","freeman-tukey","mod-log-likelihood","cressie-read"]))
            
        
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
                
        self.levels_x=0
        self.levels_y=0
        self.levels_z=0
            
    
    def check_value(self,indice,liste_variables):
        if isinstance(indice,int):
            if  indice<len(self.df.columns) and indice>=0:
                if self.learner:
                    return self.learner.nameFromId(indice)
                else:
                    return indice
            else:
                raise ValueError("index is supposed included between 0 and number of variables")
        elif isinstance(indice,str):
            if indice in liste_variables:
                if self.learner:
                    return indice
                else:
                    return liste_variables.index(indice)
            else:
                print("non exisintg index is ", indice)
                raise ValueError ("You try to condition on a not existing value")            
        else:
            raise TypeError("Format expected for condition test is either a string or the integer index of the variable")
    
    def column_treatment(self):
        column_x=self.df.iloc[:,self.ind_x].values
        column_y=self.df.iloc[:,self.ind_y].values
        
        #convert each factor as a class of integers
        #role of compression
        le = preprocessing.LabelEncoder()
        column_x=le.fit_transform(column_x)
        column_y=le.fit_transform(column_y)
        
        #create a dataframe combining the 3 columns
        if self.ind_z:
            column_z=np.array(self.df.iloc[:,self.ind_z[0]].values,dtype=str)
            self.levels_z=len(np.unique(column_z))
            
            for indice in self.ind_z[1:]:               
                column_z=np.array([x1 + x2  for x1,x2 in zip(column_z,np.array(self.df.iloc[:,indice].values,dtype=str))])
                self.levels_z+=len(self.df.iloc[:,indice].unique())
            condition_df = {'X':column_x,'Y':column_y,'Z':le.fit_transform(column_z)}
            condition_df=pd.DataFrame.from_dict(condition_df)
            #compute nlevels of x, y and z
            self.levels_x,self.levels_y=len(np.unique(column_x)),len(np.unique(column_y))
        else:
            condition_df = {'X':column_x,'Y':column_y,'Z':np.zeros(len(column_y),dtype='int')}
            condition_df=pd.DataFrame.from_dict(condition_df)
            #create an empty column of the same factor of Z, in order to reduce size of code, and use a general method
            self.levels_x,self.levels_y,self.levels_z=len(np.unique(column_x)),len(np.unique(column_y)),1
        
        return condition_df
        
    def apply_heuristic_one(self):        
        return len(self.df.index) < (self.power_rule * self.levels_x * self.levels_y * self.levels_z)   
        
    def compute_statistic(self,group,adjust_degree=False,total_df=None,permut=False): 
        #if permutation is true, we first shuffle values before creating contingency table        
        if permut:
            group=group.copy()
            group['Y']=np.random.permutation(group['Y'].values)    
        effectif_observe_by_z=np.array(pd.crosstab(group['X'],group['Y']))    
        chi2_stat_by_z, p, dof, theoric_effectif =stats.chi2_contingency(effectif_observe_by_z, correction=False, lambda_=self.calculation_method)
        
                
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
    
        
    
    def classic_test(self,condition_df):      
        #we sum statistic for each group Z        
        statistic=condition_df.groupby(['Z']).agg(self.compute_statistic)['X'].sum()
        dof=(self.levels_x-1)*(self.levels_y-1)*self.levels_z
        p_value=1-stats.chi2.cdf(statistic,dof)        
        return (statistic,p_value,dof)     
       
        
    def adjusted_test(self,condition_df): 
        groups,adjusted_df,chi2=condition_df.groupby(['Z']),(self.levels_x-1)*(self.levels_y-1)*self.levels_z,0       
        for  name,group in groups:            
            chi_temp,adjusted_df=self.compute_statistic(group,adjust_degree=True,total_df=adjusted_df)
            chi2+=chi_temp            
        
        p_value=1-stats.chi2.cdf(chi2,adjusted_df)
        return (chi2,p_value,adjusted_df)
         
        
    def permutation(self,condition_df,semi_parametric_option=False):        
        #first compute chi2 statistic in the original database
        chi2_original=self.classic_test(condition_df)[0]
        #store stats for each permuation
        chi2_permut_vector=np.empty(self.permutation_number)
        #conditional permutation test
        
        for permutation in range (self.permutation_number):                          
            chi2_permut=0            
            for  name,group in condition_df.groupby(['Z']):  
               chi2_permut+=self.compute_statistic(group,permut=True)
            
            chi2_permut_vector[permutation]=chi2_permut                
               
        if semi_parametric_option:
            df_adjusted=np.mean(chi2_permut_vector)            
            return (chi2_original, 1-stats.chi2.cdf(chi2_original,df_adjusted,df_adjusted))
        else:
            return (chi2_original,np.count_nonzero(chi2_original<chi2_permut_vector)/self.permutation_number,None)
        
    
    
    def heuristic_permutation(self,condition_df):
        chi2_0, dof_0, p_0=self.classic_test(condition_df)     
        #apply heuristic one 
        if (p_0 >0.5 or p_0<0.001):
            return (chi2_0, dof_0, p_0)
        #store stats for each permutation
        chi2_permut_vector=np.array([])
        
        for permutation in range (self.permutation_number):  
            
            chi2_permut=0            
            for  name,group in condition_df.groupby(['Z']):                     
               chi2_permut+=self.compute_statistic(group,permut=True)               
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
      condition_df=self.column_treatment()           
      #several useful imports to make type transition between Python and R
       
      #pandas2ri uses a library required to convert Python dataframe to R dataframes
      from rpy2.robjects import r, pandas2ri
      pandas2ri.activate()
      r_dataframe=pandas2ri.py2ri(condition_df)
      
      r('''
        # execute ci.test
        require (bnlearn)
        f <- function(x,y ,z=c(), df,test) {
        
        #convert dataframe as factor df
        for(i in 1:ncol(df)){
        df[,i] <- as.factor(df[,i])
        }
        #two types of tests, according whether z is present or not    
        if (is.null(z)) { resultat=ci.test(x=x,y=y,data=df[,c(1,2)],test=test) }
        else {resultat=ci.test(x=x,y=y,z=z,data=df,test=test) }                
        return (c(as.numeric(resultat$statistic),as.numeric(resultat$p.value)))                
        }
        ''')
      ci_test = r['f']
      #condtionnal test
      
      if self.ind_z:
          return ci_test('X','Y','Z',df=r_dataframe, test=type_test)
      #classic independance test
      else:          
          return ci_test(x='X',y='Y',df=r_dataframe, test=type_test)
        
        
    def semi_parametric(self,condition_df):
        return self.permutation(condition_df, semi_parametric_option=True)
    
    def realize_test(self): 
        if self.verbosity:
            print("Statistic test carried out is {} with degrees adjustement {} and following threshold value {}".format(self.calculation_method, self.dof_adjustment, self.threshold_pvalue))
        if self.learner:           
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
            condition_df=self.column_treatment()            
            #apply heuristic one of power rule            
            if self.power_rule and self.apply_heuristic_one():
                retour=math.inf,1,None            
            retour=type_test[self.dof_adjustment](condition_df)
            if self.verbosity:
                print("Computed values are, in that order, stat computed: {}, pvalue: {} and degrees of freedom: {}".format(*retour))
        return retour 
    
    
    
    
    
    
    def testIndepFromChi2(self,p_value=None):
        """
        Just prints the resultat of the independance test
        """  
        if p_value:
            
            variables=list(self.df.columns)
        if p_value>=self.threshold_pvalue:
            if self.ind_z:
                print("From Chi2 tests, is '{}' indep from '{}' given {} : {}".format(variables[self.ind_x],variables[self.ind_y],[variables[x] for x in self.ind_z],p_value))
            else:
                print("From Chi2 tests, is '{}' indep from '{}' : {}".format(variables[self.ind_x],variables[self.ind_y],p_value)) 
        
    
    
        
        
        
def compute_independance_tests(bn,sizes,test,nb_test=20,**dico_independance):
    """
    Using $nb_test$ database of size $size$ from the bn $bn$, 
    computing the p-value for a list $lindep$ of conditional independence tests, using $test$ type.
    """
    #pvalue vector stores pvalues for a given size for several conditions
    pvalue_vector=[]
    
    #pvalue amplitude stores the shift between max and min pvalue measured
    pvalue_min_error,p_value_max_error=[],[]
    
    for size in sizes:  
        print("la taille actuelle est de ", size)         
        pvalue_temp=np.empty(nb_test)
        for indice in range(nb_test):
            gum.generateCSV(bn,os.path.join("databases","sample_score.csv"),size,False)
            df=pd.read_csv(os.path.join("databases","sample_score.csv"))
            pvalue_temp[indice]=indepandance(df,*test,**dico_independance).realize_test()[1]
        print("le vecteur temp est ", pvalue_temp)
        pvalue_vector.append(np.mean(pvalue_temp))
        pvalue_min_error.append((np.mean(pvalue_temp)-np.min(pvalue_temp)))
        p_value_max_error.append((np.max(pvalue_temp)-np.mean(pvalue_temp)))
    return pvalue_vector,np.array([pvalue_min_error,p_value_max_error])

def plot_independance_tests(bn,sizes,lindep,nb_test=20,**dico_independance):

    fig = plt.figure()
    for test in lindep:
        print("we study {} test".format(format_test(test)))
        pvalues,pvalues_err=compute_independance_tests(bn,sizes,test,nb_test,**dico_independance)
        plt.errorbar(sizes, pvalues, yerr=pvalues_err, uplims=True, lolims=True, label=format_test(test))
    #at the end of computation, delete temporay file created
    os.remove(os.path.join("databases","sample_score.csv"))
    plt.tick_params(rotation=90)
    plt.xlabel ("data size")
    plt.ylabel ("pValue")
    plt.legend(bbox_to_anchor=(0.15, 0.88, 0.7, .102), loc=3,ncol=3, mode="expand", borderaxespad=0.)
    plt.title("{} test using {} technique".format(dico_independance.get("calculation_method","pearson"), \
                                                     dico_independance.get("dof_adjustment","classic")))
    fig.tight_layout()
    plt.savefig("resultat_independance_R.png")

def format_test(test):
    if test[2]:
        return ("{} indep {} given {}.".format(*test))
    else:
        return ("{} indep {}.".format(*test[0:2]))


import time

def controler_temps(nb_secs):
    """Contrôle le temps mis par une fonction pour s'exécuter.
    Si le temps d'exécution est supérieur à nb_secs, on affiche une alerte"""
    
    def decorateur(fonction_a_executer):
        """Notre décorateur. C'est lui qui est appelé directement LORS
        DE LA DEFINITION de notre fonction (fonction_a_executer)"""
        
        def fonction_modifiee():
            """Fonction renvoyée par notre décorateur. Elle se charge
            de calculer le temps mis par la fonction à s'exécuter"""
            
            tps_avant = time.time() # Avant d'exécuter la fonction
            valeur_renvoyee = fonction_a_executer() # On exécute la fonction
            tps_apres = time.time()
            tps_execution = tps_apres - tps_avant
            print("le temps d'ex est de ", tps_execution)
            return valeur_renvoyee
        return fonction_modifiee
    return decorateur

   
if __name__ == "__main__":     
    
    asia_bn=gum.loadBN(os.path.join("true_graphes_structures","asia.bif"))   
    """
    lindep=[("asia","smoke",['lung']),    ("asia","smoke",[]),
                                          ("dysp","smoke",[]),
                                          ("dysp","smoke",["lung","bronc"]),
                                          ("tub","bronc",[]),
                                          ("tub","bronc",["dysp"])]
    """
    lindep=[("asia","smoke",['lung'])]
    sizes=[20000,50000,100000,200000]
    #plot_independance_tests(asia_bn,sizes,lindep,nb_test=20,R_test=True)
    #gnb.showBN(asia_bn,size="20")
    
    """
    ie=gum.LazyPropagation(asia_bn)
    ie.addJointTarget({"asia"})
    ie.makeInference()
    ie.jointPosterior({"asia"}).toarray()
    """
    
    df_test = pd.DataFrame({'X': ["foo", "bar", "foo", "foo", "bar", "bar"], 'Y': ["one", "two", "one", "two", "one","two"],'Z1': ["dull", "shiny", "shiny", "dull", "shiny","shiny"],'Z2': ["hello","hello","hello","hello","mince","mince"]})
    df_test.to_csv("df_test.csv")                 
    #gum.generateCSV(asia_bn,"sample_asia.csv",100,False)
    learner=gum.BNLearner("df_test.csv")
    #avec py agrum
    print(indepandance(df_test,"X","Y",["Z1","Z2"],learner=learner).realize_test())
    #avec R
    print(indepandance(df_test,"X","Y",["Z1","Z2"],R_test=True).realize_test())
    #avec own program
    print(indepandance(df_test,"X","Y",["Z1","Z2"]).realize_test())
    
    #avec pre traitement df
    df_traited=indepandance(df_test,"X","Y",["Z1","Z2"]).column_treatment()
    df_traited.to_csv("df_traited.csv")
    learner=gum.BNLearner("df_traited.csv")
    #avec py agrum
    print(indepandance(df_traited,"X","Y",["Z"],learner=learner).realize_test())
    
    
    
   

    
    
    
    
    
    
    
    
    
    
   
    
    