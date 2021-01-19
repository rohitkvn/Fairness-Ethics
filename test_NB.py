from sklearn.naive_bayes import MultinomialNB
from Preprocessing import preprocess
from Postprocessing import *
from utils import *

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

NBC = MultinomialNB()
NBC.fit(training_data, training_labels)

training_class_predictions = NBC.predict_proba(training_data)
training_predictions = []
test_class_predictions = NBC.predict_proba(test_data)
test_predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i][1])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i][1])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

training_race_cases, thresholds = enforce_single_threshold(training_race_cases)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

# ADD MORE PRINT LINES HERE - THIS ALONE ISN'T ENOUGH
# YOU NEED ACCURACY AND COST FOR TRAINING AND TEST DATA
# PLUS WHATEVER RELEVANT METRICS ARE USED IN YOUR POSTPROCESSING METHOD, TO ENSURE EPSILON WAS ENFORCED
print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("Accuracy on test data:")
print(get_total_accuracy(test_race_cases))
print("")

print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

print("Cost on test data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")

print("Attempting to enforce max profit...TEST_DATA")
#train_race_cases, thresholds = enforce_predictive_parity(training_race_cases,0.01)
if test_race_cases is not None:
    print("")
    for group in test_race_cases.keys():
	    num_positive_predictions = get_num_predicted_positives(test_race_cases[group])
	    prob = num_positive_predictions / len(test_race_cases[group])
	    print("Probability of positive prediction for " + str(group) + ": " + str(prob))


    print("")
    for group in test_race_cases.keys():
	    accuracy = get_num_correct(test_race_cases[group]) / len(test_race_cases[group])
	    print("Accuracy for " + group + ": " + str(accuracy))

    print("")
    for group in test_race_cases.keys():
	    FPR = get_false_positive_rate(test_race_cases[group])
	    print("FPR for " + group + ": " + str(FPR))

    print("")
    for group in test_race_cases.keys():
	    FNR = get_false_negative_rate(test_race_cases[group])
	    print("FNR for " + group + ": " + str(FNR))

    print("")
    for group in test_race_cases.keys():
	    TPR = get_true_positive_rate(test_race_cases[group])
	    print("TPR for " + group + ": " + str(TPR))

    print("")
    for group in test_race_cases.keys():
	    TNR = get_true_negative_rate(test_race_cases[group])
	    print("TNR for " + group + ": " + str(TNR))

    print("")
    for group in thresholds.keys():
	    print("Threshold for " + group + ": " + str(thresholds[group]))

print("Attempting to enforce max profit...TRAIN_DATA")
#training_race_cases, thresholds = enforce_predictive_parity(training_race_cases,0.01)
if training_race_cases is not None:
    print("")
    for group in training_race_cases.keys():
	    num_positive_predictions = get_num_predicted_positives(training_race_cases[group])
	    prob = num_positive_predictions / len(training_race_cases[group])
	    print("Probability of positive prediction for " + str(group) + ": " + str(prob))


    print("")
    for group in training_race_cases.keys():
	    accuracy = get_num_correct(training_race_cases[group]) / len(training_race_cases[group])
	    print("Accuracy for " + group + ": " + str(accuracy))

    print("")
    for group in training_race_cases.keys():
	    FPR = get_false_positive_rate(training_race_cases[group])
	    print("FPR for " + group + ": " + str(FPR))

    print("")
    for group in training_race_cases.keys():
	    FNR = get_false_negative_rate(training_race_cases[group])
	    print("FNR for " + group + ": " + str(FNR))

    print("")
    for group in training_race_cases.keys():
	    TPR = get_true_positive_rate(training_race_cases[group])
	    print("TPR for " + group + ": " + str(TPR))

    print("")
    for group in training_race_cases.keys():
	    TNR = get_true_negative_rate(training_race_cases[group])
	    print("TNR for " + group + ": " + str(TNR))

    print("")
    for group in thresholds.keys():
	    print("Threshold for " + group + ": " + str(thresholds[group]))




