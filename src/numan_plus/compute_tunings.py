import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev, sqrt
import scipy.stats as stats

def average_tuning_curves(Q, H):
    #Q: array of type of stimulus (trials,)
    #H: matrix of responses (trials, cells)
    Qrange = np.unique(Q) #Qrange: array of unique type of stimuli (n_numerosities,)
    tuning_curves = np.array([H[Q==j,:].mean(axis=0) for j in Qrange]) # (n_num, cells)
    
    return tuning_curves

def preferred_numerosity(Q, H):
    tuning_curves = average_tuning_curves(Q, H)

    #pref_num = np.unique(Q)[np.argmax(tuning_curves, axis=0)]
    # added abs to consider possible inhibitions!!!!
    pref_num = np.unique(Q)[np.argmax(np.abs(tuning_curves), axis=0)]

    # Find if activity is excitatory (positive) or inhibitory (negative)
    max_values = tuning_curves[np.argmax(np.abs(tuning_curves), axis=0), np.arange(tuning_curves.shape[1])]
    excitatory_or_inhibitory = np.where(max_values > 0, 'excitatory', 'inhibitory')
    
    return pref_num, excitatory_or_inhibitory

def get_tuning_matrix(Q, R, pref_num, excitatory_or_inhibitory, n_numerosities):
    # 1. Calculate average tuning curve of each unit
    tuning_curves = average_tuning_curves(Q, R)
    
    # Arrays to store results for excitatory and inhibitory neurons
    tuning_mat_exc = []
    tuning_mat_inh = []
    tuning_err_exc = []
    tuning_err_inh = []

    # 2. Calculate population tuning curves separately for excitatory and inhibitory neurons
    for q in np.arange(n_numerosities):
        # For excitatory neurons
        exc_indices = np.logical_and(pref_num == q, excitatory_or_inhibitory == 'excitatory')
        if np.any(exc_indices):
            tuning_mat_exc.append(np.mean(tuning_curves[:, exc_indices], axis=1))
            tuning_err_exc.append(np.std(tuning_curves[:, exc_indices], axis=1) / np.sqrt(np.sum(exc_indices)))
        else:
            tuning_mat_exc.append(np.zeros(tuning_curves.shape[0]))  # Append zeros if no excitatory neurons found
            tuning_err_exc.append(np.zeros(tuning_curves.shape[0]))

        # For inhibitory neurons
        inh_indices = np.logical_and(pref_num == q, excitatory_or_inhibitory == 'inhibitory')
        if np.any(inh_indices):
            tuning_mat_inh.append(np.mean(tuning_curves[:, inh_indices], axis=1))
            tuning_err_inh.append(np.std(tuning_curves[:, inh_indices], axis=1) / np.sqrt(np.sum(inh_indices)))
        else:
            tuning_mat_inh.append(np.zeros(tuning_curves.shape[0]))  # Append zeros if no inhibitory neurons found
            tuning_err_inh.append(np.zeros(tuning_curves.shape[0]))

    # Convert lists to numpy arrays
    tuning_mat_exc = np.array(tuning_mat_exc)
    tuning_err_exc = np.array(tuning_err_exc)
    tuning_mat_inh = np.array(tuning_mat_inh)
    tuning_err_inh = np.array(tuning_err_inh)

    # 3. Normalize population tuning curves to the 0-1 range for both excitatory and inhibitory neurons
    def normalize_tuning(tuning_mat, tuning_err):
        tmmin = tuning_mat.min(axis=1)[:, None]
        tmmax = tuning_mat.max(axis=1)[:, None]
        tuning_mat_norm = (tuning_mat - tmmin) / (tmmax - tmmin)
        tuning_err_norm = tuning_err / (tmmax - tmmin)
        return tuning_mat_norm, tuning_err_norm

    tuning_mat_exc, tuning_err_exc = normalize_tuning(tuning_mat_exc, tuning_err_exc)
    tuning_mat_inh, tuning_err_inh = normalize_tuning(tuning_mat_inh, tuning_err_inh)

    return tuning_mat_exc, tuning_err_exc, tuning_mat_inh, tuning_err_inh

def plot_tunings(tuning_mat, tuning_err, n_numerosities, colors_list, save_path=None, save_name=None):
    # Plot population tuning curves on linear scale
    Qrange = np.arange(n_numerosities)
    plt.figure(figsize=(9,4))
    plt.title(save_name)
    plt.subplot(1,2,1)
    for i, (tc, err) in enumerate(zip(tuning_mat, tuning_err)):
        plt.errorbar(Qrange, tc, err, color=colors_list[i])
        plt.xticks(ticks=Qrange, labels=np.arange(n_numerosities))
    plt.xlabel('Numerosity')
    plt.ylabel('Normalized Neural Activity')
    # Plot population tuning curves on log scale
    plt.subplot(1,2,2)
    for i, (tc, err) in enumerate(zip(tuning_mat, tuning_err)):
        plt.errorbar(np.arange(n_numerosities), tc, err, color=colors_list[i]) # offset x axis by one to avoid taking the log of zero
    plt.xscale('log', base=2)
    #plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(ticks=Qrange, labels=Qrange)
    plt.xlabel('Numerosity')
    plt.ylabel('Normalized Neural Activity')
    # save figure
    if not (save_name is None):
        if not (save_path is None):
            plt.savefig(save_path + '/'+ save_name + '.svg')
            plt.savefig(save_path + '/'+ save_name + '.png', dpi=900)
        else:
            plt.savefig(save_name + '.svg')
            plt.savefig(save_name + '.png', dpi=900)
    plt.show()

def plot_selective_cells_histo(pref_num, n_numerosities, colors_list, chance_lev=None, save_path = None, save_name=None):
    Qrange = np.arange(n_numerosities)
    hist = [np.sum(pref_num==q) for q in Qrange]
    perc  = hist/np.sum(hist)

    # plot number neurons percentages and absolute distance tuning
    plt.figure(figsize=(4,4))
    plt.bar(Qrange, hist, width=0.8, color=colors_list)
    for x, y, p in zip(Qrange, hist, perc):
        plt.text(x, y, str(y)+'\n'+str(round(p*100,1))+'%')
    if not (chance_lev is None):
        plt.axhline(y=chance_lev, color='k', linestyle='--')
    plt.xticks(np.arange(n_numerosities),np.arange(n_numerosities).tolist())
    plt.xlabel('Preferred Numerosity')
    plt.ylabel('Number of cells')
    plt.title(save_name)
    # save figure
    if not (save_name is None):
        if not (save_path is None):
            plt.savefig(save_path + '/'+ save_name + '.svg')
            plt.savefig(save_path + '/'+ save_name + '.png', dpi=900)
        else:
            plt.savefig(save_name + '.svg')
            plt.savefig(save_name + '.png', dpi=900)
    plt.show()

def abs_dist_tunings(tuning_mat, n_numerosities, absolute_dist=0, save_path = None, save_name=None):
    if absolute_dist == 1:
        distRange = np.arange(n_numerosities).tolist()
    else:
        distRange = np.arange(-(n_numerosities-1),n_numerosities).tolist()
    dist_tuning_dict = {}
    for i in distRange:
        dist_tuning_dict[str(i)]=[]
    for pref_n in np.arange(n_numerosities):
        for n in np.arange(n_numerosities):
            if absolute_dist == 1:
                dist_tuning_dict[str(abs(n - pref_n))].append(tuning_mat[pref_n][n])
            else:
                dist_tuning_dict[str(n - pref_n)].append(tuning_mat[pref_n][n])
            
    dist_avg_tuning = [mean(dist_tuning_dict[key]) for key in dist_tuning_dict.keys()]
    dist_err_tuning = []
    for key in dist_tuning_dict.keys():
        if len(dist_tuning_dict[key])>1:
            dist_err_tuning.append(np.nanstd(dist_tuning_dict[key])/sqrt(len(dist_tuning_dict[key])))
        else:
            dist_err_tuning.append(0)

    #plot
    plt.figure(figsize=(4,4))
    if not np.isnan(dist_avg_tuning).all():
        plt.errorbar(distRange, dist_avg_tuning, dist_err_tuning, color='black')
    plt.xticks(distRange)
    plt.xlabel('Absolute numerical distance')
    plt.ylabel('Normalized Neural Activity')
    plt.title(save_name)
    # save figure
    if not (save_name is None):
        if not (save_path is None):
            plt.savefig(save_path + '/'+ save_name + '.svg')
            plt.savefig(save_path + '/'+ save_name + '.png', dpi=900)
        else:
            plt.savefig(save_name + '.svg')
            plt.savefig(save_name + '.png', dpi=900)
            
    plt.show()
    
    #statistics t-test comparisons
    for i in distRange:
        if i==5:
            continue
        print('Comaparison absolute distances '+ str(i)+ ' and '+ str(i+1)+ ':')
        print(stats.ttest_ind(a=dist_tuning_dict[str(i)], b=dist_tuning_dict[str(i+1)], equal_var=False))
    
    return dist_avg_tuning, dist_err_tuning
