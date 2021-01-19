from sklearn import svm
from Preprocessing import preprocess
from Postprocessing import *
from utils import *
from Report_Results import report_results

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

np.random.seed(42)
SVR = svm.LinearSVR(C=1.0/float(len(test_data)), max_iter=5000)
SVR.fit(training_data, training_labels)


predictions_train = SVR.predict(training_data)
predictions_test = SVR.predict(test_data)


training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, predictions_train, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, predictions_test, test_labels)

training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases,0.01)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])


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

print("-----------------------------------------------------------------")
print("Attempting to enforce equal opportunity...TEST_DATA")
if test_race_cases is not None:
    print("")
    for group in test_race_cases.keys():
	    num_positive_predictions = get_num_predicted_positives(test_race_cases[group])
	    prob = num_positive_predictions / len(test_race_cases[group])
	    print("Probability of positive prediction for " + str(group) + ": " + str(prob))

    print("")
    for group in test_race_cases.keys():
        PPV = get_positive_predictive_value(test_race_cases[group])
        print("PPV for " + group + ": " + str(PPV))

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
    print("")

print("-----------------------------------------------------------------")
print("Attempting to enforce equal opportunity...TRAIN DATA")
if training_race_cases is not None:
    print("")
    for group in training_race_cases.keys():
	    num_positive_predictions = get_num_predicted_positives(training_race_cases[group])
	    prob = num_positive_predictions / len(training_race_cases[group])
	    print("Probability of positive prediction for " + str(group) + ": " + str(prob))


    print("")
    for group in training_race_cases.keys():
        PPV = get_positive_predictive_value(training_race_cases[group])
        print("PPV for " + group + ": " + str(PPV))
    
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
print("-----------------------------------------------------------------")