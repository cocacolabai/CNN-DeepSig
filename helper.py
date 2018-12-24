from fasta_reader import fastaer
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def readdata(filename, maxlen, dataset='train'):
    X = []
    Y = []
    with open(filename, 'r') as f:
        for ID, seq, label in fastaer(f, dataset):
            if dataset == 'train':
                b = 2 * [0]
                if 'S' in label:
                    b[0] = 1
                else:
                    b[1] = 1
                Y.append(b)
            X.append(seq2input(seq.replace('U', 'C'), maxlen))

    if dataset == 'train':
        return np.array(pad_sequences(X, padding='post', maxlen=maxlen)), np.array(Y)
    else:
        return np.array(pad_sequences(X, padding='post', maxlen=maxlen))

def seq2input(seq, maxlen):
    aa_order = 'VLIMFWYGAPSTCHRKQEND'
    # aa_order = 'ARNDCQEGHILKMFPSTWYV'
    # aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    water_like = 'GYNQSTC'
    anti_water = 'AVLIFWMP'
    other = 'DHEKR'
    maxlen_20 = []
    for alphabet in seq[:maxlen]:
        m_20 = [0.0]*20
        try:
            i = aa_order.index(alphabet)
            if alphabet in water_like:
                m_20[i] = 2.0
            elif alphabet in anti_water:
                m_20[i] = -2.0
            else:
                m_20[i] = 1.0
            # m_20[i] = 1.0
        except ValueError:
            pass
        maxlen_20.append(m_20)
    return maxlen_20

def strip_newline(filename):
    with open(filename, 'r') as f:
        with open(filename + '_out', 'w') as w:
            before = f.readline().strip()
            while before is not None:
                seq = str()
                if '>' in before:
                    w.write(before + '\n')
                    while True:
                        after = f.readline().strip()
                        if '>' in after:
                            before = after
                            break
                        elif not after: # terminate
                            before = None
                            break
                        seq += after
                    w.write(seq + '\n')

def get_result(pred_prob, group):
    # get index 0(SP) or 1(other)
    p_pred = np.argmax(pred_prob, axis=1)

    print(group)
    for index, class_ in enumerate(p_pred):
        if class_ == 0 and pred_prob[index][0] < 0.72:
            print(pred_prob[index][0])
            p_pred[index] = 1

    NC = np.sum(p_pred)
    SP = p_pred.shape[0] - NC

    return SP, NC

def avg_acu(filename):
    """
        Get average MCC from twenty times.

        filename: filename that store twenty times MCC result
    """
    sum_ = 0.0
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            sum_ += float(line)
    print(sum_/20)


def check_train_data(filename, maxlen, dataset='train'):
    X = []
    Y = []
    count = dict()
    with open(filename, 'r') as f:
        for ID, seq, label in fastaer(f, dataset):
            if dataset == 'train':
                b = 2 * [0]
                if 'S' in label:
                    b[0] = 1
                    for a in seq[:label.count('S')]:
                        if a in count.keys():
                            count[a] = count[a] + 1
                        else:
                            count[a] = 1
                else:
                    b[1] = 1
                Y.append(b)
            X.append(seq2input(seq.replace('U', 'C'), maxlen))
    print(sorted([(key, count[key]) for key in count], key=lambda x: x[1], reverse=True))

    if dataset == 'train':
        return np.array(pad_sequences(X, padding='post', maxlen=maxlen)), np.array(Y)
    else:
        return np.array(pad_sequences(X, padding='post', maxlen=maxlen))
