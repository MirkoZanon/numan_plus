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

def plot_tuning_curves(tuning_mat_exc, tuning_err_exc,  colors=None, tuning_mat_inh=None, tuning_err_inh=None, excitatory_or_inhibitory=None):
    # Number of types of stimuli (should be the same for both matrices)
    n_stimuli = tuning_mat_exc.shape[0]  # This should match tuning_mat_inh if provided

    # Check if the color list is provided
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']  # Default colors

    # Create a single plot for both categories if both are provided
    if excitatory_or_inhibitory is not None:
        # Create a figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot for excitatory neurons
        for i in range(n_stimuli):
            axs[0].plot(np.arange(n_stimuli), tuning_mat_exc[i], color=colors[i], label=f'{i}')
            axs[0].fill_between(np.arange(n_stimuli), 
                                tuning_mat_exc[i] - tuning_err_exc[i], 
                                tuning_mat_exc[i] + tuning_err_exc[i], 
                                color=colors[i], alpha=0.2)  # Error bars
        axs[0].set_title('Tuning Curves - Excitatory Neurons')
        axs[0].set_xlabel('Numerosity')
        axs[0].set_ylabel('Response')

        # Plot for inhibitory neurons
        if tuning_mat_inh is not None and tuning_err_inh is not None:
            for i in range(n_stimuli):
                axs[1].plot(np.arange(n_stimuli), tuning_mat_inh[i], color=colors[i], label=f'{i}')
                axs[1].fill_between(np.arange(n_stimuli), 
                                    tuning_mat_inh[i] - tuning_err_inh[i], 
                                    tuning_mat_inh[i] + tuning_err_inh[i], 
                                    color=colors[i], alpha=0.2)  # Error bars
            axs[1].set_title('Tuning Curves - Inhibitory Neurons')
            axs[1].set_xlabel('Numerosity')
            axs[1].set_ylabel('Response')

        # Adjust layout and show legend
        plt.tight_layout()
        plt.show()

    else:
        # If no classification provided, plot only excitatory neurons
        plt.figure(figsize=(10, 5))
        for i in range(n_stimuli):
            plt.plot(np.arange(n_stimuli), tuning_mat_exc[i], color=colors[i], label=f'{i}')
            plt.fill_between(np.arange(n_stimuli), 
                             tuning_mat_exc[i] - tuning_err_exc[i], 
                             tuning_mat_exc[i] + tuning_err_exc[i], 
                             color=colors[i], alpha=0.2)  # Error bars
        plt.title('Tuning Curves - Excitatory Neurons')
        plt.xlabel('Numerosity')
        plt.ylabel('Response')
        plt.legend(title='Stimuli')
        plt.show()
    
def plot_abs_dist_tunings(tuning_mat_exc, n_numerosities, tuning_mat_inh=None, save_file=None, print_stats=True):
    # Define distance ranges for both absolute distance options
    distRange_abs_0 = np.arange(-(n_numerosities-1), n_numerosities).tolist()
    distRange_abs_1 = np.arange(n_numerosities).tolist()

    # Dictionaries to hold tuning values for each distance
    dist_tuning_dict_abs_0_exc = {str(i): [] for i in distRange_abs_0}
    dist_tuning_dict_abs_1_exc = {str(i): [] for i in distRange_abs_1}
    
    # If inhibitory neurons are provided, prepare their dictionaries
    dist_tuning_dict_abs_0_inh = {str(i): [] for i in distRange_abs_0} if tuning_mat_inh is not None else None
    dist_tuning_dict_abs_1_inh = {str(i): [] for i in distRange_abs_1} if tuning_mat_inh is not None else None

    # Populate the dictionaries with tuning values for excitatory neurons
    for pref_n in np.arange(n_numerosities):
        for n in np.arange(n_numerosities):
            dist_tuning_dict_abs_0_exc[str(n - pref_n)].append(tuning_mat_exc[pref_n][n])
            dist_tuning_dict_abs_1_exc[str(abs(n - pref_n))].append(tuning_mat_exc[pref_n][n])
            if tuning_mat_inh is not None:
                dist_tuning_dict_abs_0_inh[str(n - pref_n)].append(tuning_mat_inh[pref_n][n])
                dist_tuning_dict_abs_1_inh[str(abs(n - pref_n))].append(tuning_mat_inh[pref_n][n])

    # Calculate average tuning and standard deviation for excitatory neurons
    dist_avg_tuning_abs_0_exc = [mean(dist_tuning_dict_abs_0_exc[key]) if dist_tuning_dict_abs_0_exc[key] else 0 for key in dist_tuning_dict_abs_0_exc.keys()]
    dist_avg_tuning_abs_1_exc = [mean(dist_tuning_dict_abs_1_exc[key]) if dist_tuning_dict_abs_1_exc[key] else 0 for key in dist_tuning_dict_abs_1_exc.keys()]
    
    dist_err_tuning_abs_0_exc = [np.nanstd(dist_tuning_dict_abs_0_exc[key]) / sqrt(len(dist_tuning_dict_abs_0_exc[key])) if len(dist_tuning_dict_abs_0_exc[key]) > 1 else 0 for key in dist_tuning_dict_abs_0_exc.keys()]
    dist_err_tuning_abs_1_exc = [np.nanstd(dist_tuning_dict_abs_1_exc[key]) / sqrt(len(dist_tuning_dict_abs_1_exc[key])) if len(dist_tuning_dict_abs_1_exc[key]) > 1 else 0 for key in dist_tuning_dict_abs_1_exc.keys()]

    # Calculate average tuning and standard deviation for inhibitory neurons
    dist_avg_tuning_abs_0_inh = [mean(dist_tuning_dict_abs_0_inh[key]) if dist_tuning_dict_abs_0_inh[key] else 0 for key in dist_tuning_dict_abs_0_inh.keys()] if tuning_mat_inh is not None else None
    dist_avg_tuning_abs_1_inh = [mean(dist_tuning_dict_abs_1_inh[key]) if dist_tuning_dict_abs_1_inh[key] else 0 for key in dist_tuning_dict_abs_1_inh.keys()] if tuning_mat_inh is not None else None

    dist_err_tuning_abs_0_inh = [np.nanstd(dist_tuning_dict_abs_0_inh[key]) / sqrt(len(dist_tuning_dict_abs_0_inh[key])) if len(dist_tuning_dict_abs_0_inh[key]) > 1 else 0 for key in dist_tuning_dict_abs_0_inh.keys()] if tuning_mat_inh is not None else None
    dist_err_tuning_abs_1_inh = [np.nanstd(dist_tuning_dict_abs_1_inh[key]) / sqrt(len(dist_tuning_dict_abs_1_inh[key])) if len(dist_tuning_dict_abs_1_inh[key]) > 1 else 0 for key in dist_tuning_dict_abs_1_inh.keys()] if tuning_mat_inh is not None else None

    # Set up the figure with a number of subplots
    num_plots = 2  # Initialize number of plots for excitatory neurons
    if tuning_mat_inh is not None:
        num_plots += 2  # Update number of plots to include inhibitory neurons
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 layout
        axs = axs.flatten()  # Flatten for easier indexing
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1x2 layout

    # Plot for excitatory neurons - numerical distance = 0
    axs[0].errorbar(distRange_abs_0, dist_avg_tuning_abs_0_exc, 
                    yerr=dist_err_tuning_abs_0_exc, 
                    color='black')
    axs[0].set_xticks(distRange_abs_0)
    axs[0].set_xlabel('Numerical Distance')
    axs[0].set_ylabel('Normalized Neural Activity')
    axs[0].set_title('Numerical Distance Tuning Curve (Excitatory)')

    # Plot for excitatory neurons - absolute distance = 1
    axs[1].errorbar(distRange_abs_1, dist_avg_tuning_abs_1_exc, 
                    yerr=dist_err_tuning_abs_1_exc, 
                    color='black')
    axs[1].set_xticks(distRange_abs_1)
    axs[1].set_xlabel('Absolute Numerical Distance')
    axs[1].set_ylabel('Normalized Neural Activity')
    axs[1].set_title('Absolute Numerical Distance Tuning Curve (Excitatory)')

    # If inhibitory neurons are provided, create their plots
    if tuning_mat_inh is not None:
        # Plot for inhibitory neurons - numerical distance = 0
        axs[2].errorbar(distRange_abs_0, dist_avg_tuning_abs_0_inh, 
                        yerr=dist_err_tuning_abs_0_inh, 
                        color='black')
        axs[2].set_xticks(distRange_abs_0)
        axs[2].set_xlabel('Numerical Distance')
        axs[2].set_ylabel('Normalized Neural Activity')
        axs[2].set_title('Numerical Distance Tuning Curve (Inhibitory)')

        # Plot for inhibitory neurons - absolute distance = 1
        axs[3].errorbar(distRange_abs_1, dist_avg_tuning_abs_1_inh, 
                        yerr=dist_err_tuning_abs_1_inh, 
                        color='black')
        axs[3].set_xticks(distRange_abs_1)
        axs[3].set_xlabel('Absolute Numerical Distance')
        axs[3].set_ylabel('Normalized Neural Activity')
        axs[3].set_title('Absolute Numerical Distance Tuning Curve (Inhibitory)')

    # Save figures
    if save_file is not None:
        plt.savefig(f'{save_file}.svg')
        plt.savefig(f'{save_file}.png', dpi=900)

    plt.tight_layout()
    plt.show()
    
    # Dynamic distance comparisons for t-tests
    distance_comparisons = [(i, i+1) for i in range(-n_numerosities + 1, n_numerosities - 1)]

    def print_t_test_table(tuning_dict_abs_0, tuning_dict_abs_1, title):
        print(f"\n{title}")
        print(f"{'Distance Pair':<15} {'t-statistic':<15} {'p-value':<10} {'df':<5}")
        print("="*50)
        for d1, d2 in distance_comparisons:
            if str(d1) in tuning_dict_abs_0 and str(d2) in tuning_dict_abs_0:
                t_stat, p_value = stats.ttest_ind(a=tuning_dict_abs_0[str(d1)], b=tuning_dict_abs_0[str(d2)], equal_var=False)
                df = len(tuning_dict_abs_0[str(d1)]) + len(tuning_dict_abs_0[str(d2)]) - 2
                print(f"{d1} vs {d2:<7} {t_stat:.2f}       {p_value:.2f}    {df}")
                
        print("\nAbsolute Numerical Distance Comparisons:")
        print(f"{'Distance Pair':<15} {'t-statistic':<15} {'p-value':<10} {'df':<5}")
        print("="*50)
        for d1, d2 in distance_comparisons:
            if str(d1) in tuning_dict_abs_1 and str(d2) in tuning_dict_abs_1:
                t_stat, p_value = stats.ttest_ind(a=tuning_dict_abs_1[str(d1)], b=tuning_dict_abs_1[str(d2)], equal_var=False)
                df = len(tuning_dict_abs_1[str(d1)]) + len(tuning_dict_abs_1[str(d2)]) - 2
                print(f"{d1} vs {d2:<7} {t_stat:.2f}       {p_value:.2f}    {df}")

    if print_stats:
        print_t_test_table(dist_tuning_dict_abs_0_exc, dist_tuning_dict_abs_1_exc, "Excitatory Neuron Comparisons")
        if tuning_mat_inh is not None:
            print_t_test_table(dist_tuning_dict_abs_0_inh, dist_tuning_dict_abs_1_inh, "Inhibitory Neuron Comparisons")
    
    # Return data necessary for reconstructing the curves
    return {
        'exc_avg_tuning_abs_0': dist_avg_tuning_abs_0_exc,
        'exc_err_tuning_abs_0': dist_err_tuning_abs_0_exc,
        'exc_avg_tuning_abs_1': dist_avg_tuning_abs_1_exc,
        'exc_err_tuning_abs_1': dist_err_tuning_abs_1_exc,
        'inh_avg_tuning_abs_0': dist_avg_tuning_abs_0_inh,
        'inh_err_tuning_abs_0': dist_err_tuning_abs_0_inh,
        'inh_avg_tuning_abs_1': dist_avg_tuning_abs_1_inh,
        'inh_err_tuning_abs_1': dist_err_tuning_abs_1_inh,
        'distRange_abs_0': distRange_abs_0,
        'distRange_abs_1': distRange_abs_1
    }