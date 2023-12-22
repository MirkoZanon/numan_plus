from SVMdataset import *
from sklearn.model_selection import LeaveOneOut
import random
import numpy as np
import copy

def split_dataset(data: SVMdataset, split_type: str = 'loo', validation_percentage: float = 0.2, n_combinations: int = 100) -> SVMdataset:
    """
    Split data in training and validation set.

    Parameters
    ----------
    data: SVMdataset object to split

    split_type: how to split the data
        'loo' = leave one out combinaitons
        'perc' = percentage of all data

    validation_percentage: percenatage of trials in the whole dataset to use at validation (ignored for split_type = 'loo', auto to 1)

    n_combinations: number of shuffled combinaitons to vary training-validation split (ignored for split_type = 'loo', auto to number of trials)
    """
    training_indeces = []
    validation_indeces = []

    if split_type == 'loo':
        loo = LeaveOneOut()
        for tr_idx, ts_idx in loo.split(data.training_signals):
            training_indeces.append(tr_idx)
            validation_indeces.append(ts_idx)

    elif split_type == 'perc':
        n_validation = int(data.get_n_training_trials()*validation_percentage)
        print(f'You assigned {n_validation} trials to validation')
        if n_validation == 0:
            raise TypeError(f'\x1B[35m\033[1mZero trials assigned to validation! This could be due to a low number of trials available: maybe better using Leave One Out -loo- option.\x1B\033[0m')
        elif n_validation <= 3:
            print(f'\033[1mLow number of trials assigned to validation: maybe better using Leave One Out -loo- option.\033[0m')
        trials = list(range(data.get_n_training_trials()))
        for i in range(n_combinations):
            shuffled = random.sample(trials, data.get_n_training_trials())
            training_indeces.append(np.array(shuffled[n_validation:]))
            validation_indeces.append(np.array(shuffled[0:n_validation]))

    data.training_indeces = training_indeces
    data.validation_indeces = validation_indeces
    data.n_split_combinations = len(training_indeces)

    return data

def _normalize_dataset(data: SVMdataset, combination: int, verbose: bool = False) -> SVMdataset:
    """
    Normalize dataset values (in range (0,1) or by z-scorring). Requires previous training/validation split

    Parameters
    ----------
        data: SVMdataset object to normalize

        combination: index of the split combination to use
            This value should be compatible with n_split_combiations. Ignored if normalization_type = all

    Return
    ----------
        Normalized SVMdataset object
    """
    # initialize new SVMdataset not to overwrite original data
    normalized_data = copy.deepcopy(data)

    # it's required to have the training validation split before normalization
    if normalized_data.n_split_combinations == 0:
        raise TypeError('\x1B[35m\033[1mRequired training/validation split before applying normalization. Use split_dataset function before!\x1B\033[0m')
    
    # check if the chosen combination is within the split compbinations range
    if combination >= normalized_data.n_split_combinations:
        raise TypeError(f'\x1B[35m\033[1mCombination index error! You generated only {data.n_split_combinations} combinations.\x1B\033[0m')

    # normalize features range in (0,1)
    elif data.normalization_type == 'normalized':
        if data.normalization_ref == 'all':
            min_v = data.training_signals.min()
            max_v = data.training_signals.max()
        elif data.normalization_ref == 'training':
            min_v = data.training_signals.iloc[data.training_indeces[combination],:].min()
            max_v = data.training_signals.iloc[data.training_indeces[combination],:].max()
        else:
            raise TypeError(f'\x1B[35m\033[1mNormalization reference subset error! You can choose between all and training!\x1B\033[0m')
        normalized_data.training_signals = (data.training_signals - min_v)/(max_v-min_v)
        if data.test_signals is not None:
            normalized_data.test_signals = (data.test_signals - min_v)/(max_v-min_v)

    # normalize features with z-score (mean 0, sigma 1)
    elif data.normalization_type == 'zscored':
        if data.normalization_ref == 'all':
            mean_v = data.training_signals.mean()
            std_v = data.training_signals.std()
        elif data.normalization_ref == 'training':
            mean_v = data.training_signals.iloc[data.training_indeces[combination],:].mean()
            std_v = data.training_signals.iloc[data.training_indeces[combination],:].std()
        else:
            raise TypeError(f'\x1B[35m\033[1mNormalization reference subset error! You can choose between all and training!\x1B\033[0m')
        normalized_data.training_signals = (data.training_signals - mean_v)/std_v
        if data.test_signals is not None:
            normalized_data.test_signals = (data.test_signals - mean_v)/std_v

    elif data.normalization_type == 'raw':
        normalized_data.training_signals = data.training_signals
        if data.test_signals is not None:
            normalized_data.test_signals = data.test_signals

    else:
        raise TypeError(f'\x1B[35m\033[1mNormalization type error! You can choose between raw, normalized and zscored!\x1B\033[0m')

    normalized_data.current_split_combination = combination

    if verbose == True:
        print(f"Note: you are using '{data.normalization_type}' normalization with reference dataset {data.normalization_ref}. Modify SVMdataset parameters if you want different normalizations.")

    return normalized_data

def move_categories_to_test(data: SVMdataset, test_categeories: List[str]) -> SVMdataset:
    """
    Split data in training and validation set.

    Parameters
    ----------
    data: SVMdataset object to split

    test_categeories: list of categories to remove from training set and assign to test set
    """
    all_data = pd.concat([data.training_signals, data.training_labels], axis=1)
    for label in test_categeories:
        signals = all_data[all_data['target_label']==label]
        labels = signals.pop('target_label')
        if data.test_signals is None:
            data.test_signals = signals
            data.test_labels = labels
        else:
            data.test_signals = data.test_signals.append(signals)
            data.test_labels = data.test_labels.append(labels)
        all_data = all_data[all_data['target_label']!=label]
    
    data.training_signals = all_data
    data.training_labels = data.training_signals.pop('target_label')