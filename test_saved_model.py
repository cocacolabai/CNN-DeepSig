from keras.models import model_from_json
from helper import readdata, get_result
import numpy as np

with open('json_model.json', 'r') as f:
    json_string = f.read()

model = model_from_json(json_string)
model.load_weights('model_weights.h5')

"""
    TP: correct prediction in the positive classes
    TN: correct prediction in the negative classes
    FP: the number of over-predictions in the signal peptide class
    FN: the number of under-predictions in the signal peptide class
"""

X_p = readdata('./dataset/test_SP.fasta_out', 96, 'test')
X_n_1 = readdata('./dataset/test_TM.fasta_out', 96, 'test')
X_n_2 = readdata('./dataset/test_NC.fasta_out', 96, 'test')

p_pred = model.predict(X_p)
n_1_pred = model.predict(X_n_1)
n_2_pred = model.predict(X_n_2)


S_S, S_N = get_result(p_pred, 0)
N_S_1, N_N_1 = get_result(n_1_pred, 1)
N_S_2, N_N_2 = get_result(n_2_pred, 2)

TP = S_S
TN = N_N_1 + N_N_2
FP = N_S_1 + N_S_2
FN = S_N


print()
print('TP:', TP,'TN:', TN,'FP:', FP,'FN:', FN)

divv = TP*TN-FP*FN
div = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
print('MCC numerator:\t\t', divv)
print('MCC denominator:\t', div)
MCC = divv / div
print('MCC: ', MCC)

with open('avg_acu.txt', 'a') as f:
    f.write(str(float(MCC)))
    f.write('\n')
