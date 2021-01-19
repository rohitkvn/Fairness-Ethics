
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: # accuracy
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
import numpy as np
from utils import *

def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}
    raceThresholdPPRDict = {}
    for race in categorical_results:  
            predLabel = categorical_results[race]
            predTuple, labelTuple = zip(*predLabel)
            pred = np.array(predTuple)
            label = np.array(labelTuple)       
            thresholdPPRDict = {}
            for current_threshold in np.linspace(0.0,1.0,101):
                predictedPositive = np.sum(pred>current_threshold)
                predictedPositiveRatio = predictedPositive/pred.size
                if predictedPositiveRatio > 0.55:
                    thresholdPPRDict[current_threshold] = predictedPositiveRatio
            raceThresholdPPRDict[race] = thresholdPPRDict
    maximumAccuracy = 0
    races = list(categorical_results.keys())
    firstrace = races[0]
    otherRace = races[1]
    thirdRace = races[2]
    fourthRace = races[3]
    thresholdPPRDict = raceThresholdPPRDict[firstrace]
    for threshold in thresholdPPRDict :
        probable_solution = {}
        predictedPositiveRatio = thresholdPPRDict[threshold]                
        probable_solution[firstrace] = threshold
        otherthresholdPPRDict = raceThresholdPPRDict[otherRace]
        for otherthreshold in otherthresholdPPRDict :
                    otherpredictedPositiveRatio = otherthresholdPPRDict[otherthreshold]
                    if abs(predictedPositiveRatio - otherpredictedPositiveRatio) < epsilon :
                        probable_solution[otherRace] = otherthreshold
                        thirdthresholdPPRDict = raceThresholdPPRDict[thirdRace]
                        for thirdthreshold in thirdthresholdPPRDict :
                                    thirdpredictedPositiveRatio = thirdthresholdPPRDict[thirdthreshold]
                                    if abs(predictedPositiveRatio - thirdpredictedPositiveRatio) < epsilon  and \
                                    abs(otherpredictedPositiveRatio - thirdpredictedPositiveRatio) < epsilon:
                                        probable_solution[thirdRace] = thirdthreshold
                                        fourththresholdPPRDict = raceThresholdPPRDict[fourthRace]
                                        for fourththreshold in fourththresholdPPRDict :
                                                    fourthpredictedPositiveRatio = fourththresholdPPRDict[fourththreshold]
                                                    if abs(predictedPositiveRatio - fourthpredictedPositiveRatio) < epsilon  and \
                                                    abs(otherpredictedPositiveRatio - fourthpredictedPositiveRatio) < epsilon and \
                                                    abs(thirdpredictedPositiveRatio - fourthpredictedPositiveRatio) < epsilon:
                                                        probable_solution[fourthRace] = fourththreshold
                                                        racePredLabel = {}
                                                        for race in categorical_results:
                                                            solutionthreshold = probable_solution[race]
                                                            predLabel = categorical_results[race]
                                                            predTuple, labelTuple = zip(*predLabel)
                                                            pred = np.array(predTuple)
                                                            predTuple = tuple(np.array(pred>solutionthreshold)*1)
                                                            racePredLabel[race] = list(tuple(zip(predTuple,labelTuple)))
                                                        accuracy = get_total_accuracy(racePredLabel)
                                                        if accuracy > maximumAccuracy :
                                                                maximumAccuracy = accuracy
                                                                demographic_parity_data = racePredLabel
                                                                thresholds = probable_solution
    return demographic_parity_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):

    thresholds = {}
    equal_opportunity_data = {}    
    raceThresholdPPRDict = {}
    cutoffacc = {'African-American' : 0.63, 'Caucasian' : 0.63, 'Hispanic': 0.59, 'Other' : 0.6}
    for race in categorical_results:            
            predLabel = categorical_results[race]
            predTuple, labelTuple = zip(*predLabel)
            pred = np.array(predTuple)
            label = np.array(labelTuple)       
            thresholdPPRDict = {}
            for current_threshold in np.linspace(0.0,1.0,901):
                predTuple = tuple(np.array(pred>current_threshold)*1)
                predLabel = list(tuple(zip(predTuple,labelTuple)))
                acc = get_num_correct(predLabel)/len(predLabel)
                if acc > cutoffacc[race] :
                    PPV = get_true_positive_rate(predLabel)
                    thresholdPPRDict[current_threshold] = PPV
            raceThresholdPPRDict[race] = thresholdPPRDict
    maximumAccuracy = 0
    races = list(categorical_results.keys())
    firstrace = races[0]
    otherRace = races[1]
    thirdRace = races[2]
    fourthRace = races[3]
    thresholdPPRDict = raceThresholdPPRDict[firstrace]
    for threshold in thresholdPPRDict :
        probable_solution = {}
        predictedPositiveRatio = thresholdPPRDict[threshold]                
        probable_solution[firstrace] = threshold
        otherthresholdPPRDict = raceThresholdPPRDict[otherRace]
        for otherthreshold in otherthresholdPPRDict :
                    otherpredictedPositiveRatio = otherthresholdPPRDict[otherthreshold]
                    if abs(predictedPositiveRatio - otherpredictedPositiveRatio) < epsilon :
                        probable_solution[otherRace] = otherthreshold
                        thirdthresholdPPRDict = raceThresholdPPRDict[thirdRace]
                        for thirdthreshold in thirdthresholdPPRDict :
                                    thirdpredictedPositiveRatio = thirdthresholdPPRDict[thirdthreshold]
                                    if abs(predictedPositiveRatio - thirdpredictedPositiveRatio) < epsilon  and \
                                    abs(otherpredictedPositiveRatio - thirdpredictedPositiveRatio) < epsilon:
                                        probable_solution[thirdRace] = thirdthreshold
                                        fourththresholdPPRDict = raceThresholdPPRDict[fourthRace]
                                        for fourththreshold in fourththresholdPPRDict :
                                                    fourthpredictedPositiveRatio = fourththresholdPPRDict[fourththreshold]
                                                    if abs(predictedPositiveRatio - fourthpredictedPositiveRatio) < epsilon  and \
                                                    abs(otherpredictedPositiveRatio - fourthpredictedPositiveRatio) < epsilon and \
                                                    abs(thirdpredictedPositiveRatio - fourthpredictedPositiveRatio) < epsilon:
                                                        probable_solution[fourthRace] = fourththreshold
                                                        racePredLabel = {}
                                                        for race in categorical_results:
                                                            solutionthreshold = probable_solution[race]
                                                            predLabel = categorical_results[race]
                                                            predTuple, labelTuple = zip(*predLabel)
                                                            pred = np.array(predTuple)
                                                            predTuple = tuple(np.array(pred>solutionthreshold)*1)
                                                            racePredLabel[race] = list(tuple(zip(predTuple,labelTuple)))
                                                        accuracy = get_total_accuracy(racePredLabel)
                                                        if accuracy > maximumAccuracy :
                                                                maximumAccuracy = accuracy
                                                                equal_opportunity_data = racePredLabel
                                                                thresholds = probable_solution
    return equal_opportunity_data, thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}
    thresholdPPRDict={}
    threshold_data = float(0)
    max_accuracy=float(0)
    for race in categorical_results.keys():
        max_accuracy=float(0)
        for thres in np.linspace(0,1,100):
            prediction=categorical_results[race]
            threshold_pred=[]
            for p in prediction:
                if p[0]>=thres:
                    threshold_pred.append((1,p[1]))
                else:
                    threshold_pred.append((0,p[1]))
            thresholdPPRDict[race]=threshold_pred
            accuracy=get_total_accuracy(thresholdPPRDict)
            if(accuracy>max_accuracy):
                thresholds[race] = thres
                max_accuracy = accuracy
    for race in categorical_results.keys():
        prediction = categorical_results[race]
        mp_data[race] = apply_threshold(prediction,thresholds[race])
    return mp_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}
    raceThresholdPPRDict = {}
    for race in categorical_results:            
            predLabel = categorical_results[race]
            predTuple, labelTuple = zip(*predLabel)
            pred = np.array(predTuple)
            label = np.array(labelTuple)       
            thresholdPPRDict = {}
            for current_threshold in np.linspace(0,1.0,101):
                predTuple = tuple(np.array(pred>current_threshold)*1)
                predLabel = list(tuple(zip(predTuple,labelTuple)))
                acc = get_num_correct(predLabel)/len(predLabel)
                if acc > 0.62 :
                    PPV = get_positive_predictive_value(predLabel)
                    thresholdPPRDict[current_threshold] = PPV
            raceThresholdPPRDict[race] = thresholdPPRDict
    maximumAccuracy = 0
    races = list(categorical_results.keys())
    firstrace = races[0]
    otherRace = races[1]
    thirdRace = races[2]
    fourthRace = races[3]
    thresholdPPRDict = raceThresholdPPRDict[firstrace]
    for threshold in thresholdPPRDict :
        probable_solution = {}
        predictedPositiveRatio = thresholdPPRDict[threshold]                
        probable_solution[firstrace] = threshold
        otherthresholdPPRDict = raceThresholdPPRDict[otherRace]
        for otherthreshold in otherthresholdPPRDict :
                    otherpredictedPositiveRatio = otherthresholdPPRDict[otherthreshold]
                    if abs(predictedPositiveRatio - otherpredictedPositiveRatio) < epsilon :
                        probable_solution[otherRace] = otherthreshold
                        thirdthresholdPPRDict = raceThresholdPPRDict[thirdRace]
                        for thirdthreshold in thirdthresholdPPRDict :
                                    thirdpredictedPositiveRatio = thirdthresholdPPRDict[thirdthreshold]
                                    if abs(predictedPositiveRatio - thirdpredictedPositiveRatio) < epsilon  and \
                                    abs(otherpredictedPositiveRatio - thirdpredictedPositiveRatio) < epsilon:
                                        probable_solution[thirdRace] = thirdthreshold
                                        fourththresholdPPRDict = raceThresholdPPRDict[fourthRace]
                                        for fourththreshold in fourththresholdPPRDict :
                                                    fourthpredictedPositiveRatio = fourththresholdPPRDict[fourththreshold]
                                                    if abs(predictedPositiveRatio - fourthpredictedPositiveRatio) < epsilon  and \
                                                    abs(otherpredictedPositiveRatio - fourthpredictedPositiveRatio) < epsilon and \
                                                    abs(thirdpredictedPositiveRatio - fourthpredictedPositiveRatio) < epsilon:
                                                        probable_solution[fourthRace] = fourththreshold
                                                        racePredLabel = {}
                                                        for race in categorical_results:
                                                            solutionthreshold = probable_solution[race]
                                                            predLabel = categorical_results[race]
                                                            predTuple, labelTuple = zip(*predLabel)
                                                            pred = np.array(predTuple)
                                                            predTuple = tuple(np.array(pred>solutionthreshold)*1)
                                                            racePredLabel[race] = list(tuple(zip(predTuple,labelTuple)))
                                                        accuracy = get_total_accuracy(racePredLabel)
                                                        if accuracy > maximumAccuracy :
                                                                maximumAccuracy = accuracy
                                                                predictive_parity_data = racePredLabel
                                                                thresholds = probable_solution
    return predictive_parity_data, thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}
    max_accuracy = float(0)
    opt_threshold = float(0)
    categorical_results_copy = {}
    threshold_accuracy_dict = {}
    for threshold_step in np.linspace(0,1,100):
        for demography in categorical_results.keys():
            pred_label_pair = categorical_results[demography]
            categorical_results_copy[demography] = apply_threshold(pred_label_pair,threshold_step)
        accuracy = get_total_accuracy(categorical_results_copy)
        if(accuracy>max_accuracy):
            opt_threshold = threshold_step
            max_accuracy = accuracy
    for demography in categorical_results.keys():
        pred_label_pair = categorical_results[demography]
        single_threshold_data[demography] = apply_threshold(pred_label_pair,opt_threshold)
        thresholds[demography] = opt_threshold
    return single_threshold_data, thresholds