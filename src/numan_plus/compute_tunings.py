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

def plot_selective_cells_histo(pref_num, n_numerosities, colors_list, excitatory_or_inhibitory=None, chance_lev=None, save_path=None, save_name=None):
    Qrange = np.arange(n_numerosities)
    
    if excitatory_or_inhibitory is None:
        # Caso originale: plottare l'istogramma di tutti i neuroni
        hist = [np.sum(pref_num == q) for q in Qrange]
        perc = hist / np.sum(hist)

        plt.figure(figsize=(4, 4))
        plt.bar(Qrange, hist, width=0.8, color=colors_list)
        for x, y, p in zip(Qrange, hist, perc):
            plt.text(x, y, str(y) + '\n' + str(round(p * 100, 1)) + '%')

        if not (chance_lev is None):
            plt.axhline(y=chance_lev, color='k', linestyle='--')

        plt.xticks(np.arange(n_numerosities), np.arange(n_numerosities).tolist())
        plt.xlabel('Preferred Numerosity')
        plt.ylabel('Number of cells')
        plt.title(save_name)

        # Save figure
        if not (save_name is None):
            if not (save_path is None):
                plt.savefig(f'{save_path}/{save_name}.svg')
                plt.savefig(f'{save_path}/{save_name}.png', dpi=900)
            else:
                plt.savefig(f'{save_name}.svg')
                plt.savefig(f'{save_name}.png', dpi=900)

        plt.show()

    else:
        # Se excitatory_or_inhibitory è fornito, plotta i tre subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Istogramma per i neuroni excitatory
        excitatory_indices = excitatory_or_inhibitory == 'excitatory'
        hist_exc = [np.sum(pref_num[excitatory_indices] == q) for q in Qrange]
        perc_exc = hist_exc / np.sum(hist_exc)

        axes[0].bar(Qrange, hist_exc, width=0.8, color=colors_list)
        for x, y, p in zip(Qrange, hist_exc, perc_exc):
            axes[0].text(x, y, str(y) + '\n' + str(round(p * 100, 1)) + '%')

        axes[0].set_title('Excitatory Neurons')
        axes[0].set_xlabel('Preferred Numerosity')
        axes[0].set_ylabel('Number of cells')
        if not (chance_lev is None):
            axes[0].axhline(y=chance_lev, color='k', linestyle='--')

        # Istogramma per i neuroni inhibitory
        inhibitory_indices = excitatory_or_inhibitory == 'inhibitory'
        hist_inh = [np.sum(pref_num[inhibitory_indices] == q) for q in Qrange]
        perc_inh = hist_inh / np.sum(hist_inh)

        axes[1].bar(Qrange, hist_inh, width=0.8, color=colors_list)
        for x, y, p in zip(Qrange, hist_inh, perc_inh):
            axes[1].text(x, y, str(y) + '\n' + str(round(p * 100, 1)) + '%')

        axes[1].set_title('Inhibitory Neurons')
        axes[1].set_xlabel('Preferred Numerosity')
        axes[1].set_ylabel('Number of cells')
        if not (chance_lev is None):
            axes[1].axhline(y=chance_lev, color='k', linestyle='--')

        # Istogramma per tutti i neuroni (come il caso originale)
        hist = [np.sum(pref_num == q) for q in Qrange]
        perc = hist / np.sum(hist)

        axes[2].bar(Qrange, hist, width=0.8, color=colors_list)
        for x, y, p in zip(Qrange, hist, perc):
            axes[2].text(x, y, str(y) + '\n' + str(round(p * 100, 1)) + '%')

        axes[2].set_title('All Neurons')
        axes[2].set_xlabel('Preferred Numerosity')
        axes[2].set_ylabel('Number of cells')
        if not (chance_lev is None):
            axes[2].axhline(y=chance_lev, color='k', linestyle='--')

        plt.tight_layout()

        # Save figure
        if not (save_name is None):
            if not (save_path is None):
                plt.savefig(f'{save_path}/{save_name}.svg')
                plt.savefig(f'{save_path}/{save_name}.png', dpi=900)
            else:
                plt.savefig(f'{save_name}.svg')
                plt.savefig(f'{save_name}.png', dpi=900)

        plt.show()

def plot_tunings_and_abs_dist(tuning_mat_exc=None, tuning_err_exc=None, 
                              tuning_mat_inh=None, tuning_err_inh=None,
                              n_numerosities=0, colors_list=None,
                              excitatory_or_inhibitory=None, log_scale=False,
                              save_path=None, save_name=None):
    """
    Plots tuning curves and absolute distance tunings for excitatory and inhibitory neurons.
    
    Parameters:
        tuning_mat_exc, tuning_err_exc: Tuning matrices and errors for excitatory neurons.
        tuning_mat_inh, tuning_err_inh: Tuning matrices and errors for inhibitory neurons.
        n_numerosities: Number of numerosities.
        colors_list: List of colors for plotting.
        excitatory_or_inhibitory: Optional array to differentiate excitatory and inhibitory neurons.
        log_scale: If True, also plots log scale.
        save_path: Path to save the plots.
        save_name: Name for saving the plot.
    """
    Qrange = np.arange(n_numerosities)
    
    # Handle log(0) issues by replacing 0 with a small value
    log_safe_Qrange = np.where(Qrange == 0, 1e-5, Qrange)

    fig, axs = plt.subplots(2, 2 if log_scale else 1, figsize=(12, 8))
    fig.suptitle(save_name)

    # Plot for excitatory neurons
    if tuning_mat_exc is not None and tuning_err_exc is not None:
        axs[0, 0].set_title('Excitatory Neurons - Linear Scale')
        for i, (tc, err) in enumerate(zip(tuning_mat_exc, tuning_err_exc)):
            axs[0, 0].errorbar(Qrange, tc, err, color=colors_list[i])
        axs[0, 0].set_xlabel('Numerosity')
        axs[0, 0].set_ylabel('Normalized Neural Activity')

        if log_scale:
            axs[0, 1].set_title('Excitatory Neurons - Log Scale')
            for i, (tc, err) in enumerate(zip(tuning_mat_exc, tuning_err_exc)):
                axs[0, 1].errorbar(log_safe_Qrange, tc, err, color=colors_list[i])
            axs[0, 1].set_xscale('log', base=2)
            axs[0, 1].set_xlabel('Numerosity')
            axs[0, 1].set_ylabel('Normalized Neural Activity')

    # Plot for inhibitory neurons
    if tuning_mat_inh is not None and tuning_err_inh is not None:
        axs[1, 0].set_title('Inhibitory Neurons - Linear Scale')
        for i, (tc, err) in enumerate(zip(tuning_mat_inh, tuning_err_inh)):
            axs[1, 0].errorbar(Qrange, tc, err, color=colors_list[i])
        axs[1, 0].set_xlabel('Numerosity')
        axs[1, 0].set_ylabel('Normalized Neural Activity')

        if log_scale:
            axs[1, 1].set_title('Inhibitory Neurons - Log Scale')
            for i, (tc, err) in enumerate(zip(tuning_mat_inh, tuning_err_inh)):
                axs[1, 1].errorbar(log_safe_Qrange, tc, err, color=colors_list[i])
            axs[1, 1].set_xscale('log', base=2)
            axs[1, 1].set_xlabel('Numerosity')
            axs[1, 1].set_ylabel('Normalized Neural Activity')

    # Plot absolute distance tunings
    dist_avg_tuning_exc, dist_err_tuning_exc = abs_dist_tunings(tuning_mat_exc, n_numerosities)
    dist_avg_tuning_inh, dist_err_tuning_inh = abs_dist_tunings(tuning_mat_inh, n_numerosities)

    plt.tight_layout()
    if save_name:
        if save_path:
            plt.savefig(f'{save_path}/{save_name}.svg')
            plt.savefig(f'{save_path}/{save_name}.png', dpi=900)
        else:
            plt.savefig(f'{save_name}.svg')
            plt.savefig(f'{save_name}.png', dpi=900)

    plt.show()

# Additional function to calculate absolute distance tunings
def abs_dist_tunings(tuning_mat, n_numerosities):
    distRange = np.arange(-(n_numerosities - 1), n_numerosities).tolist()
    dist_tuning_dict = {str(i): [] for i in distRange}
    
    for pref_n in np.arange(n_numerosities):
        for n in np.arange(n_numerosities):
            dist_tuning_dict[str(n - pref_n)].append(tuning_mat[pref_n][n])

    dist_avg_tuning = [np.mean(dist_tuning_dict[key]) for key in dist_tuning_dict.keys()]
    dist_err_tuning = [np.std(dist_tuning_dict[key]) / np.sqrt(len(dist_tuning_dict[key]))
                       if len(dist_tuning_dict[key]) > 1 else 0 for key in dist_tuning_dict.keys()]

    return dist_avg_tuning, dist_err_tuning