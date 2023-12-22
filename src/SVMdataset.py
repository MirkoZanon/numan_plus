from typing import Union
from typing import List
import pandas as pd

class SVMdataset:

    signals = pd.DataFrame # the activity values (Trials X Freatures)
    labels = pd.DataFrame # category label for each trial
    training_indeces = List[List[int]] # indeces of training trials (list for different combinations as in validation_indeces)
    validation_indeces = List[List[int]] # indeces of validation trials (list for different combinations as in training_indices)
    n_split_combinations = int # number of different training/validation split combinations 
    current_split_combination = int # index of the split the normalized data refers to (if raw data, -1)
    normalization_type = str # type of normalization to apply: 'raw' no normalization, 'norm' (0,1) normalizaiton, 'zscore' z-scoring
    normalization_ref = str # where to calculate normalization parameters: 'all' entire dataset, 'training' only training subset

    def __init__(self, 
                 training_data: Union[pd.DataFrame, str], 
                 training_labels: Union[pd.DataFrame, str] = None, 
                 test_data: Union[pd.DataFrame, str] = None, 
                 test_labels: Union[pd.DataFrame, str] = None, 
                 labels_index: int = 0, 
                 trials_axis: int = 0, 
                 normalization_type: str ='raw', 
                 normalization_ref: str = 'all'):
        """
        Initialize SVM dataset based on a pandas dataframe input, to provide the standard data matrix shaped as Trials X Features

        Parameters
        ----------
        training_data: the original dataset used for training and validate SVM, provided as pandas dataframe. 
            If 'training_labels' is not provided 'training_data' is supposed to include the target category labels for each trial (specify the relative index in 'labels_index').
        
        training_labels: the target category labels for each trial.
            It is required if 'data' doesn't include this information. If this parameter is provided labels_index will be ignored.
            
        labels_index: index of the column (or row) where the category labels for each trial are (index follows python notation, starting from 0).
            If trials_axis = 0 label_index will refer to a column, otherwise to a row.

        trials_axis: axis along which the different trials are reported. 0 = trials along rows (default); 1 = trials along columns.

        normalization_type: 'raw' no normalization, 'normalized', 'zscored'

        normalization_ref: 'all'

        """
        ### training dataset

        # save training data matrix as pandas dataframe
        if type(training_data) == str:
            if training_data.endswith('csv'):
                pandas_data = pd.read_csv(training_data,index_col=None)
                if "Unnamed: 0" in pandas_data:
                    pandas_data.drop(columns="Unnamed: 0", inplace=True)
            elif training_data.endswith('xlsx') | training_data.endswith('xlx'):
                pandas_data = pd.read_excel(training_data,index_col=None)
                if "Unnamed: 0" in pandas_data:
                    pandas_data.drop(columns="Unnamed: 0", inplace=True)
            else:
                raise RuntimeError('File type provided not supported! Please provide the name of a csv or excel file for training set')
        elif type(training_data) == pd.DataFrame:
            pandas_data = training_data
        else:
            raise RuntimeError('Data type not recognized! Please provide the name of a csv/excel file or a pandas dataframe for training set')

        # orient matrix to fit Trials x Features
        if trials_axis != 0:
            pandas_data = pandas_data.T

        ### test dataset

        # save test data matrix as pandas dataframe
        if test_data is None:
            self.test_signals = None
            self.test_labels = None
        elif type(test_data) == str:
            if test_data.endswith('csv'):
                pandas_data_test = pd.read_csv(test_data,index_col=None)
                if "Unnamed: 0" in pandas_data_test:
                    pandas_data_test.drop(columns="Unnamed: 0", inplace=True)
            elif test_data.endswith('xlsx') | test_data.endswith('xlx'):
                pandas_data_test = pd.read_excel(test_data,index_col=None)
                if "Unnamed: 0" in pandas_data_test:
                    pandas_data_test.drop(columns="Unnamed: 0", inplace=True)
            else:
                raise RuntimeError('File type provided not supported! Please provide the name of a csv or excel file for test set')
        elif type(test_data) == pd.DataFrame:
            pandas_data_test = test_data
        else:
            raise RuntimeError('Data type not recognized! Please provide the name of a csv/excel file or a pandas dataframe for test set')

        # orient matrix to fit Trials x Features
        if (test_data is not None) & (trials_axis != 0):
            pandas_data_test = pandas_data_test.T

        ### training labels

        # save optional labels in Dataset_SVM object
        if type(training_labels) == str:
            if training_labels.endswith('csv'):
                self.training_labels = pd.read_csv(training_labels,index_col=None)
                if "Unnamed: 0" in pandas_data:
                    pandas_data.drop(columns="Unnamed: 0", inplace=True)
            elif training_labels.endswith('xlsx') | training_labels.endswith('xls'):
                self.training_labels = pd.read_excel(training_labels,index_col=None)
                if "Unnamed: 0" in pandas_data:
                    pandas_data.drop(columns="Unnamed: 0", inplace=True)
            else:
                raise RuntimeError('File type provided not supported! Please provide the name of a csv or excel file')
        elif type(training_labels) == pd.DataFrame:
            self.training_labels = training_labels
        elif training_labels is not None:
            raise RuntimeError('Data type not recognized! Please provide the name of a csv/excel file or a pandas dataframe')

        ### test labels

        # save optional test labels in Dataset_SVM object
        if type(test_labels) == str:
            if training_labels.endswith('csv'):
                self.test_labels = pd.read_csv(test_labels,index_col=None)
                if "Unnamed: 0" in pandas_data_test:
                    pandas_data_test.drop(columns="Unnamed: 0", inplace=True)
            elif test_labels.endswith('xlsx') | test_labels.endswith('xls'):
                self.test_labels = pd.read_excel(test_labels,index_col=None)
                if "Unnamed: 0" in pandas_data_test:
                    pandas_data_test.drop(columns="Unnamed: 0", inplace=True)
            else:
                raise RuntimeError('File type provided not supported! Please provide the name of a csv or excel file for test labels')
        elif type(test_labels) == pd.DataFrame:
            self.test_labels = test_labels
        elif test_labels is not None:
            raise RuntimeError('Data type not recognized! Please provide the name of a csv/excel file or a pandas dataframe for test labels')


        # extract labels from data matrix if not provided separately and save in Dataset_SVM object
        if training_labels is None:
            pandas_data.rename(columns={pandas_data.columns[labels_index]: "target_label"}, inplace=True)
            self.training_labels = pandas_data.pop('target_label')
            # ensure that labels are strings
            self.training_labels = self.training_labels.astype(str)
        if (test_labels is None) & (test_data is  not None):
            pandas_data_test.rename(columns={pandas_data_test.columns[labels_index]: "target_label"}, inplace=True)
            self.test_labels = pandas_data_test.pop('target_label')
            self.test_labels = self.test_labels.astype(str)
        
        # save dataframe in Dataset_SVM object
        self.training_signals = pandas_data
        if test_data is not None:
            self.test_signals = pandas_data_test

        # set number of split combinations to 0, current_split to -1 (ref to raw data), normalization to raw data
        self.n_split_combinations = 0
        self.current_split_combination = -1
        self.normalization_type = normalization_type
        self.normalization_ref = normalization_ref
        self.real_target_training = None
        self.predicted_target_training = None
        self.real_target_validation = None
        self.predicted_target_validation = None
        self.real_target_test = None
        self.predicted_target_test = None

        # print object properties
        if test_data is not None:
            print(f'You stored a dataset with {self.training_signals.shape[1]} features, {self.training_signals.shape[0]} training trials and {self.test_signals.shape[0]} test trials.')
        else:
            print(f'You stored a dataset with {self.training_signals.shape[1]} features and {self.training_signals.shape[0]} training trials. No test set defined, only validation (subsetting training set) will be performed.')
        print(f'You have {len(self.training_labels.unique())} category labels: {self.training_labels.unique()}.')
        if normalization_type == 'raw':
            print('No normalizaiton is applied to your data')
        else:
            print(f'From now on, {normalization_type} data on {normalization_ref} trials will be used.')

    def get_n_training_trials(self) -> int:
        """
        Return the overall number of trials in the dataset
        """
        return self.training_signals.shape[0]
    
    def get_n_features(self) -> int:
        """
        Return the overall number of features in the dataset
        """
        return self.training_signals.shape[1]
    
    def get_n_labels(self) -> int:
        """
        Return the overall number of categories in the dataset
        """
        return len(self.training_labels.unique())
    
    def get_unique_labels(self) -> List[str]:
        """
        Return a list with the different categories in the dataset
        """
        return self.training_labels.unique()
    
    def get_n_trials_per_labels(self) -> pd.DataFrame:
        """
        Return a dataframe with the the different categories and the corresponding number of trials
        """
        return self.training_labels.value_counts()
    
    def get_n_split_combinations(self) -> pd.DataFrame:
        """
        Return the number of different combinations in which the dataset is split between training and validation
        """
        if self.training_indeces is None:
            print('No split was performed! Please split dataset in training/validation by using split_dataset function')
        return len(self.training_indeces)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def set_normalization_type(self, type: str):
        """
        Modify normalization type in 'normalization_type'
        """
        if type in ['raw', 'normalized', 'zscored']:
            self.normalization_type = type
        else:
            raise RuntimeError('Normalization type not recognized! Choose between raw, norm or zscore')
        
    def set_normalization_subset(self, subset: str):
        """
        Modify normalization subset in 'normalization_ref'
        """
        if type in ['all', 'training']:
            self.normalization_ref = subset
        else:
            raise RuntimeError('Normalization subset not recognized! Choose between all or train')
