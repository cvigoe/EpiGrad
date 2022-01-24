import collections

def calculate_auc(FPRs, TPRs):
    auc = 0
    for index, fpr in enumerate(FPRs):
        if index == 0:
            continue
        auc += (fpr - FPRs[index-1])*TPRs[index-1]
    return abs(auc)

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)