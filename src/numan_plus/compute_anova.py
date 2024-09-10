import numpy as np
import numpy.typing as npt
import pandas as pd
from statistics import mean, stdev, sqrt
import scipy.stats as stats
from tqdm.notebook import tqdm
import tifffile as tif
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

from numan_plus import compute_tunings

# Vectorized implementation of two-way ANOVA
def anova_two_way(A, B, Y):
    num_cells = Y.shape[1]
    
    A_levels = np.unique(A); a = len(A_levels)
    B_levels = np.unique(B); b = len(B_levels)
    Y4D = np.array([[Y[(A==i)&(B==j)] for j in B_levels] for i in A_levels])
    
    r = Y4D.shape[2]

    Y = Y4D.reshape((-1, Y.shape[1]))
    
    # only test cells (units) that are active (gave a nonzero response to at least one stimulus) to avoid division by zero errors
    active_cells = np.where(np.abs(Y).max(axis=0)>0)[0]
    Y4D = Y4D[:,:,:,active_cells]
    Y = Y[:, active_cells]
    
    N = Y.shape[0]
    
    Y_mean = Y.mean(axis=0)
    Y_mean_A = Y4D.mean(axis=1).mean(axis=1)
    Y_mean_B = Y4D.mean(axis=0).mean(axis=1)
    Y_mean_AB = Y4D.mean(axis=2)

    
    SSA = r*b*np.sum((Y_mean_A - Y_mean)**2, axis=0)
    SSB = r*a*np.sum((Y_mean_B - Y_mean)**2, axis=0)
    SSAB = r*((Y_mean_AB - Y_mean_A[:,None] - Y_mean_B[None,:] + Y_mean)**2).sum(axis=0).sum(axis=0)
    SSE = ((Y4D-Y_mean_AB[:,:,None])**2).sum(axis=0).sum(axis=0).sum(axis=0)
    SST = ((Y-Y_mean)**2).sum(axis=0)

    DFA = a - 1; DFB = b - 1; DFAB = DFA*DFB
    DFE = (N-a*b); DFT = N-1
    
    MSA = SSA / DFA
    MSB = SSB / DFB
    MSAB = SSAB / DFAB
    MSE = SSE / DFE
    
    FA = MSA / MSE
    FB = MSB / MSE
    FAB = MSAB / MSE
    
    pA = np.nan*np.zeros(num_cells)
    pB = np.nan*np.zeros(num_cells)
    pAB = np.nan*np.zeros(num_cells)
    
    pA[active_cells] = stats.f.sf(FA, DFA, DFE)
    pB[active_cells] = stats.f.sf(FB, DFB, DFE)
    pAB[active_cells] = stats.f.sf(FAB, DFAB, DFE)
    
    return pA, pB, pAB, FA, FB, FAB


def anova_two_way_permutations(A, B, Y, num_perm, show_progress=True):
    a, b, c, FA0, FB0, FAB0 = anova_two_way(A, B, Y)
    num_cells = Y.shape[1]

    A_levels = np.unique(A); a = len(A_levels)
    B_levels = np.unique(B); b = len(B_levels)
    Y4D = np.array([[Y[(A==i)&(B==j)] for j in B_levels] for i in A_levels])
    
    r = Y4D.shape[2]

    Y = Y4D.reshape((-1, Y.shape[1]))
    
    active_cells = np.where(np.abs(Y).max(axis=0)>0)[0]
    Y4D = Y4D[:,:,:,active_cells]
    Y = Y[:, active_cells]

    FA0 = np.expand_dims(FA0, axis=1)
    FB0 = np.expand_dims(FB0, axis=1)
    FAB0 = np.expand_dims(FAB0, axis=1)
    nperm = num_perm
    FA = np.nan*np.zeros((active_cells.shape[0], nperm))
    FB = np.nan*np.zeros((active_cells.shape[0], nperm))
    FAB = np.nan*np.zeros((active_cells.shape[0], nperm))

    perm_iter = range(nperm)
    if show_progress:
        perm_iter = tqdm(perm_iter, desc='Permutations')
    
    for i in perm_iter:
        np.random.shuffle(Y)
        a, b, c, FA[:,i], FB[:,i], FAB[:,i] = anova_two_way(A, B, Y)

    pA = np.nan*np.zeros(num_cells)
    pB = np.nan*np.zeros(num_cells)
    pAB = np.nan*np.zeros(num_cells)

    pA[active_cells] = np.sum(np.greater_equal(FA, FA0), axis=1) / nperm
    pB[active_cells] = np.sum(np.greater_equal(FB, FB0), axis=1) / nperm
    pAB[active_cells] = np.sum(np.greater_equal(FAB, FAB0), axis=1) / nperm

    return pA, pB, pAB
def group_logical_and(group1, group2):
    """
    Performs element-wise logical and
    on two boolean arrays expects 1D or 2D (with second dimention of size 1) boolean arrays.
    Returns a 1D boolean array.
    """
    if len(group1.shape) >1: 
        assert len(group1.shape)==2, "expects less than 2 dimension but got more than 2 dimension"
        group1 = np.squeeze(group1)

    if len(group2.shape) >1: 
        assert len(group2.shape) ==2, "expects less than 2 dimension but got more than 2 dimension"
        group2 = np.squeeze(group2)

    assert group1.shape == group2.shape , "Dimentions of things for logical and must match"

    return np.logical_and(group1,group2)


def get_peristim(experiment, timepoints_to_use:list, stimulus_type:tuple, signals:npt.NDArray)->npt.NDArray: # add signals to argument
    """
    Makes a 3d array (timepoints, peristimulus cycle, cells) 
    from experiment_truncated_drift_corrected.db and json signal
    Args:
        timepoints_to_use: list, grabs indexed around the stimulus, +/- is before and after
        stimulus_type: tuple, get the stimulus you want from experiment_truncated_drift_corrected
            All of a certain number or shape
        signals: numpy array, the signal trace
    Returns:
        3D Numpy array of (timepoints, peristimulus cycle, cells)
    """
    idx = experiment.choose_volumes(stimulus_type)
    #make empty list
    idx_block = []

    #makes a list of indexes to get signal later
    idx_block = [i + j for i in idx for j in timepoints_to_use]

    #make a list of signals for each stimulus
    #order will be (timepoints, peristimulus cycle, cells)
    number_per_peristim = len(timepoints_to_use)
    number_of_peristim_cycle = int(len(idx_block)/number_per_peristim)
    #signal is a numpy array and downstream stuff are too so dn blocks will be too
    block_signal = np.empty((len(idx_block), signals.shape[1]))

    #print(f'Note: you are exctracting signal from {signals.shape[1]} neurons')
    # for foo in np.arange(signals.shape[1]):
    for neur in np.arange(signals.shape[1]):
        for t_point in np.arange(len(idx_block)):
            block_signal[t_point, neur] = signals[idx_block[t_point], neur]
    # print(block_signal.shape)

    #reshape so we can easily find the mean later
    #block_signal_reshaped = block_signal.reshape(number_per_peristim, number_of_peristim_cycle, signals.shape[1])
    block_signal_reshaped = block_signal.reshape(number_of_peristim_cycle, number_per_peristim, signals.shape[1])
    return block_signal_reshaped

def ANOVA_preprocess (experiment, stim_signal_exact, brain_region_tag, stim_volumes, signals):
    """
    Crate following variables for ANOVA calculation:
    Hf: matrix of responses (cells X trials); Q: array of type of stimulus (trials,); C: array of type of control condition (trials,)
    The responses are evaluated as avg across 3 time points from stimulus onset
    """

    ## take a single avg value as reference response to stimulus (avg across 3 vols from stimulation)
    stim_signal_prov = np.zeros((stim_signal_exact.shape[0],stim_signal_exact.shape[1],3))
    stim_signal_prov[:,:,0] = stim_signal_exact
    for i in [1,2]: # add additional volume after stimulus to calculate final avg signal
        stim_add_volumes = [x+i for x in stim_volumes]
        stim_add_signal = signals[:,stim_add_volumes]
        stim_signal_prov[:,:,i] = stim_add_signal
    ## create final matrix Hf for anova (cells X trials)
    stim_signal = stim_signal_prov.mean(axis=2)
    print('You are passing to the ANOVA, (cells,trials): ' + str(stim_signal.shape)+'\n')
    annotation_dict2= {f"cell_{ic}": stim_signal[ic] for ic in np.arange(len(signals))}
    annotation_dict=experiment.get_volume_annotations(stim_volumes)
    annotation_dict.update(annotation_dict2)
    annotation_df=pd.DataFrame(annotation_dict)
    Hf = np.array(annotation_df.iloc[:, 4:annotation_df.shape[1]])
    #print('Final dataset, shape -> (trials,cells): ' + str(Hf.shape))

    #calculate control trials label array C for anova (trials,): in our case array of 0,1,2,3,4,5 corresponding to the 6 combinations of getical control conditions (shape*spread)
    C_pd = pd.factorize((annotation_df['shape']+ annotation_df['spread']), sort=True)
    C = np.array(C_pd[0])
    #print('Control conditions label array, shape -> (trials,): ' + str(C.shape))

    #calculate stimulus trials label array Q for anova (trials,): in our case array of 0,1,2,3,4 corresponding to the 5 numerosity
    Q_pd = pd.factorize(annotation_df['number'], sort=True)
    Q = np.array(Q_pd[0])
    #print('Stimulus label array, shape -> (trials,): ' + str(Q.shape))

    return Hf, C, Q


def compute_anova_neurons(Q, C, Hf, alpha_level, n_permutations, filtered_idx):#, brain_region_tag, save_df):
    print('\nRunning permutation ANOVA on real dataset:')

    # Find numorosity selective units (anova_cells) using a two-way ANOVA with permutations (permute data and check F distribution for p-value)
    pN, pC, pNC = anova_two_way_permutations(Q, C, Hf, n_permutations, show_progress=True)
    anova_cells = np.where((pN<alpha_level) & (pNC>alpha_level) & (pC>alpha_level))[0]
    R = Hf[:,anova_cells]
    #save_df['Total segmented'] = [Hf.shape[1]]
    #save_df['Anova selective'] = [R.shape[1]]

    chance_lev = Hf.shape[1]*alpha_level/6
    #save_df['chance n cells per group'] = [chance_lev]
    #print('Chance number of cells for group: '+str(chance_lev))
    print('Number of anova cells = %i (%0.2f%%)'%(len(anova_cells), 100*len(anova_cells)/Hf.shape[1]))

    ##Creating dictionary for plotting anova cells: 'cell_ID'=preferred numerosity
    pref_num, excitatory_or_inhibitory = compute_tunings.preferred_numerosity(Q, R)
    number_cells_dic = {'anova_cells': filtered_idx[anova_cells], 'pref_num': pref_num, 'excitatory_or_inhibitory': excitatory_or_inhibitory}
    anova_df = pd.DataFrame.from_dict(data=number_cells_dic)
    #for n in range(6):
    #    save_df[f'Preferring_{n}'] = [sum(pref_num==n)]
    #
    #os.makedirs('./caiman_final_datasets', exist_ok=True) 
    #anova_df.to_csv(f'./caiman_final_datasets/numerosityCells_{brain_region_tag}.csv')
    #print('\033[1m\nYour number units are calculated.\033[0m\nYou can find them in ./processed/caiman_final_dataset')

    return anova_cells, R, chance_lev, anova_df, pref_num, excitatory_or_inhibitory#, save_df

def compute_shuffled_anova_neurons(Q, C, Hf, alpha_level, n_permutations, show_progress=False):
    if show_progress:
        print('\nRunning permutation ANOVA on shuffled dataset:')

    # Shuffle the datasets
    Q_S = shuffle(Q, random_state=0)
    C_S = shuffle(C, random_state=0)

    # Perform two-way ANOVA with permutations
    pN_s, pC_s, pNC_s = compute_anova.anova_two_way_permutations(Q_S, C_S, Hf, n_permutations, show_progress)
    
    # Identify selective units based on p-values
    anova_cells_shuffled = np.where((pN_s < alpha_level) & (pNC_s > alpha_level) & (pC_s > alpha_level))[0]
    R_S = Hf[:, anova_cells_shuffled]

    if show_progress:
        print('Number of ANOVA cells on random shuffled data = %i (%0.2f%%)' % 
              (len(anova_cells_shuffled), 100 * len(anova_cells_shuffled) / Hf.shape[1]))

    # Compute preferred numerosity for the selected neurons
    pref_num_shuffled, excitatory_or_inhibitory_shuffled = compute_tunings.preferred_numerosity(Q_S, R_S)

    return anova_cells_shuffled, Q_S, C_S, R_S, pref_num_shuffled, excitatory_or_inhibitory_shuffled

def run_multiple_shuffles(Q, C, Hf, alpha_level, n_permutations, n_reps=1000, show_inner_progress=False):
    n_numerosities = 6  # Define the number of preferred numerosity levels
    n_neurons_shuffled_excitatory = np.zeros((n_numerosities, n_reps))
    n_neurons_shuffled_inhibitory = np.zeros((n_numerosities, n_reps))
    n_neurons_shuffled_total = np.zeros((n_numerosities, n_reps))

    for i in tqdm(range(n_reps), desc='Shuffling repetitions'):
        anova_cells_shuffled, _, _, R_S, pref_num_shuffled, excitatory_or_inhibitory_S = compute_shuffled_anova_neurons(
            Q, C, Hf, alpha_level, n_permutations, show_inner_progress)

        # Get the types of selected neurons directly
        selected_neuron_types = excitatory_or_inhibitory_S
        selected_preferred_numerosities = pref_num_shuffled

        # Count excitatory and inhibitory neurons for each preferred numerosity level
        for q in range(n_numerosities):
            count_exc = np.sum(selected_preferred_numerosities[selected_neuron_types == 'excitatory'] == q)
            count_inh = np.sum(selected_preferred_numerosities[selected_neuron_types == 'inhibitory'] == q)
            count_total = np.sum(selected_preferred_numerosities == q)

            n_neurons_shuffled_excitatory[q, i] = count_exc
            n_neurons_shuffled_inhibitory[q, i] = count_inh
            n_neurons_shuffled_total[q, i] = count_total

    # Calculate the mean and SEM for both neuron types and total
    chance_means_excitatory = np.mean(n_neurons_shuffled_excitatory, axis=1)
    chance_sems_excitatory = np.std(n_neurons_shuffled_excitatory, axis=1) / np.sqrt(n_reps)

    chance_means_inhibitory = np.mean(n_neurons_shuffled_inhibitory, axis=1)
    chance_sems_inhibitory = np.std(n_neurons_shuffled_inhibitory, axis=1) / np.sqrt(n_reps)

    chance_means_total = np.mean(n_neurons_shuffled_total, axis=1)
    chance_sems_total = np.std(n_neurons_shuffled_total, axis=1) / np.sqrt(n_reps)

    chance_means = {
        'excitatory': chance_means_excitatory,
        'inhibitory': chance_means_inhibitory,
        'total': chance_means_total
    }
    
    chance_sems = {
        'excitatory': chance_sems_excitatory,
        'inhibitory': chance_sems_inhibitory,
        'total': chance_sems_total
    }

    for q in range(n_numerosities):
        print(f"Preferred Numerosity {q}:")
        print(f"  Excitatory chance level: {int(round(chance_means_excitatory[q]))} ± {int(round(chance_sems_excitatory[q]))}")
        print(f"  Inhibitory chance level: {int(round(chance_means_inhibitory[q]))} ± {int(round(chance_sems_inhibitory[q]))}")
        print(f"  Total chance level: {int(round(chance_means_total[q]))} ± {int(round(chance_sems_total[q]))}")

    return chance_means, chance_sems

# CREATE ANOVA CELLS TIF VOLUMES TO VISUALIZE 3D CELLS DISTRIBUTION
def save_anova_mask(spots, spot_tag, region_tag, vol_size, resolution):
    # vol_size and resolution ordered as: (Z, Y, X)
    print('Creating tiff with cells in 3D volume...')
    anova_group_tags = ["anova_groups", "anova_num_0", "anova_num_1","anova_num_2","anova_num_3","anova_num_4","anova_num_5"]
    for anova_tag in anova_group_tags:

        r = spots.groups[region_tag]
        a = spots.groups[anova_tag]
        combined_group = group_logical_and(r, a)    
        mask = spots.get_group_mask(combined_group, vol_size) #spots.groups[combined_tag], (Z, Y, X))

        tif.imwrite(f'./spots/masks/mask_from_{spot_tag}_{region_tag}_{anova_tag}.tif',
                                mask.astype(np.uint16), shape=vol_size,
                                metadata={'spacing': resolution[0], 'unit': 'um', 'axes': 'ZYX'},
                                resolution=(1 / resolution[1], 1 / resolution[2]), imagej=True)


def save_anova_spots(spots, anova_df, spot_tag):

    print('Saving spots with neurons info...')

    anova_cell = anova_df["anova_cells"].values
    pref_num = anova_df["pref_num"].values.astype(int)
    #create anova group
    anova_groups = np.zeros(spots.num_spots)
    anova_groups[anova_cell] = 1
    #create group numerosity 0
    anova_num_0 = np.zeros(spots.num_spots)
    anova_num_0[anova_cell[pref_num == 0]] = 1
    #create group numerosity 1
    anova_num_1 = np.zeros(spots.num_spots)
    anova_num_1[anova_cell[pref_num == 1]] = 1
    #create group numerosity 2
    anova_num_2 = np.zeros(spots.num_spots)
    anova_num_2[anova_cell[pref_num == 2]] = 1
    #create group numerosity 3
    anova_num_3 = np.zeros(spots.num_spots)
    anova_num_3[anova_cell[pref_num == 3]] = 1
    #create group numerosity 4
    anova_num_4 = np.zeros(spots.num_spots)
    anova_num_4[anova_cell[pref_num == 4]] = 1
    #create group numerosity 5
    anova_num_5 = np.zeros(spots.num_spots)
    anova_num_5[anova_cell[pref_num == 5]] = 1


    spots.add_groups({"anova_groups":anova_groups,
                    "anova_num_0":anova_num_0,
                    "anova_num_1":anova_num_1, 
                    "anova_num_2":anova_num_2, 
                    "anova_num_3":anova_num_3, 
                    "anova_num_4":anova_num_4, 
                    "anova_num_5":anova_num_5}, 
                    rewrite=True)

    spots.to_json(f"./spots/signals/spots_{spot_tag}.json")
