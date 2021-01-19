from numpy.random import seed
seed(4940)
from tensorflow import set_random_seed
set_random_seed(80)

import numpy as np
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from Preprocessing import preprocess
from Postprocessing import *
from utils import *

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)


activation = "relu"
model = Sequential()
model.add(Dense(len(metrics)*2, activation=activation, kernel_regularizer=regularizers.l2(0.1), input_shape = (len(metrics), )))
model.add(Dense(30, activation=activation, kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss="binary_crossentropy")
model.fit(training_data, training_labels, epochs=30, batch_size=300, validation_data=(test_data, test_labels), verbose=0)

data = np.concatenate((training_data, test_data))
labels = np.concatenate((training_labels, test_labels))

predictions_train = model.predict(training_data)
predictions_train = np.squeeze(predictions_train, axis=1)

predictions_test = model.predict(test_data)
predictions_test = np.squeeze(predictions_test, axis=1)

#for i in range(len(training_labels)):
#    training_predictions.append(predictions_train[i][1])

#for i in range(len(test_labels)):
#    test_predictions.append(predictions_test[i][1])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, predictions_train, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, predictions_test, test_labels)

training_race_cases, thresholds = enforce_predictive_parity(training_race_cases,0.01)

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
