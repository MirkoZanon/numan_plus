from SVMdataset import *
import AdjustSplit
from sklearn import svm
import sklearn.metrics as metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from itertools import product
from tqdm.contrib.itertools import product

def runSVM(data: SVMdataset, kernel: str = 'rbf', C: int = 1, gamma: int = None, shuffle: bool = False, verbose: bool = False):
    if shuffle == True:
            print('ATTENTION: You are running the analysis with randomly shuffled data labels!')

    y_real_training = []
    y_real_validation = []
    y_real_test = []
    y_pred_training = []
    y_pred_validation = []
    y_pred_test = []


    if verbose == True:
        print(f'You are running {data.n_split_combinations} combinations')
    for i in range(data.n_split_combinations):
        #create svm classifier
        norm_data = AdjustSplit._normalize_dataset(data, i, verbose=False)

        if gamma is None:
            gamma = 1/norm_data.get_n_features()/np.array(norm_data.training_signals).var()
        model = svm.SVC(kernel=kernel, C = C, gamma = gamma) #C=regression parameter, kernel=linear/poly/rbf(default)/sigmoid/precomuted

        
        #train model
        target_labels = norm_data.training_labels.iloc[norm_data.training_indeces[i],]
        if shuffle == True:
            target_labels = target_labels.sample(frac = 1)
        model.fit(norm_data.training_signals.iloc[norm_data.training_indeces[i],], target_labels)

        #predict for training data
        y_real_training.extend(target_labels)
        y_pred_training.extend(model.predict(norm_data.training_signals.iloc[norm_data.training_indeces[i],]))

        #predict for validation data
        y_real_validation.extend(norm_data.training_labels.iloc[norm_data.validation_indeces[i],])
        y_pred_validation.extend(model.predict(norm_data.training_signals.iloc[norm_data.validation_indeces[i],]))

        #predict for test data
        if data.test_signals is not None:
            y_real_test.extend(norm_data.test_labels)
            y_pred_test.extend(model.predict(norm_data.test_signals))
            
    accuracy_training = metrics.accuracy_score(y_real_training, y_pred_training)
    accuracy_validation = metrics.accuracy_score(y_real_validation, y_pred_validation)

    if verbose == True:
        print(f'You used {kernel} kernel, C={C:.2f}, gamma={gamma:.2f}, obtaining an accuracy of {accuracy_training:.2f} at training and {accuracy_validation:.2f} at validation.')

    data.real_target_training = y_real_training
    data.predicted_target_training = y_pred_training
    data.real_target_validation = y_real_validation
    data.predicted_target_validation = y_pred_validation
    if data.test_signals is not None:
        data.real_target_test = y_real_test
        data.predicted_target_test = y_pred_test

    return list(y_real_training), list(y_real_validation), list(y_real_test), list(y_pred_training), list(y_pred_validation), list(y_pred_test)

def optimizeParameters(data: SVMdataset, C_list:List[float] = [1], gamma_list:List[float] = [1], verbose: bool = False):
    best_accuracy_training = 0
    best_accuracy_validation = 0
    best_kernel = None
    best_C = None,
    best_gamma = None

    k_list = ['poly', 'sigmoid', 'rbf', 'linear']
    for param in product(k_list,C_list, gamma_list):
        y_real_training, y_real_validation, y_pred_training, y_pred_validation = runSVM(data, kernel = param[0], C = param[1], gamma = param[2], verbose = verbose)
        accuracy_training = metrics.accuracy_score(y_real_training, y_pred_training)
        accuracy_validation = metrics.accuracy_score(y_real_validation, y_pred_validation)
        if accuracy_validation >= best_accuracy_validation:
            best_accuracy_training = accuracy_training
            best_accuracy_validation = accuracy_validation
            best_kernel = param[0]
            best_C = param[1]
            best_gamma = param[2]
        
    print(f'You varied parameters within: kernel={k_list}, C={C_list}, gamma={gamma_list}')
    print(f'You obtain the best accuracy of {best_accuracy_training} at training and {best_accuracy_validation} at validation, with kernel {best_kernel}, C={best_C} and gamma={best_gamma}.')

    return best_kernel, best_C, best_gamma

def printConfusionMatrix(data: SVMdataset, type: List[str], order_labels: List = None, title: str = 'Confusion matrix', normalized: bool = True):
    if order_labels is None:
        labels = data.get_unique_labels()
    else:
        labels = order_labels
        ### add check comparing original labels with provided ones, to be sure they are the same

    fig, ax = plt.subplots(1,len(type), figsize=(7*len(type),5))
    cms = [pd.DataFrame(0, index=labels, columns=labels) for _ in type]
    for it, t in enumerate(type):
        if t == 'training':
            real_labels = data.real_target_training
            predicted_labels = data.predicted_target_training
        elif t == 'validation':
            real_labels = data.real_target_validation
            predicted_labels = data.predicted_target_validation
        elif t == 'test':
            real_labels = data.real_target_test
            predicted_labels = data.predicted_target_test
        else:
            print('Confusion matrix type errore! You can choose between training, validation and test')

        if real_labels is None:
            continue
        else:
            for il,l in enumerate(predicted_labels):
                cms[it][l].loc[real_labels[il]] += 1
            
            if normalized:
                # normalize cm values in (0,1)
                for index_label, row in cms[it].iterrows():
                    normalized_row = cms[it].loc[index_label,]/(cms[it].loc[index_label,].sum())
                    cms[it].loc[index_label,:] = normalized_row
                    #confusion matrix
                sns.heatmap(cms[it], annot=True, fmt=".0%", ax=ax[it], vmin=0, vmax=1)
            else:
                #confusion matrix
                sns.heatmap(cms[it], annot=True, fmt=".0%", ax=ax[it])

            if it == 0:
                ax[it].set_ylabel('True labels')
            ax[it].set_xlabel('Prediction labels')
            ax[it].set_title(f'Confusion matrix - {t} set')

    plt.show()
    return cms