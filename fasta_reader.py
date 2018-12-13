"""
    fd: file descriptor (file object)
"""

def fastaer(fd, dataset='train'):
    while True:
        label = None

        ID = fd.readline().strip()
        seq = fd.readline().strip()
        if dataset == 'train':
            label = fd.readline().strip()

        if not ID:
            break

        yield ID, seq, label
