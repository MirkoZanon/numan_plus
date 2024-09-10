import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean, stdev, sqrt
import scipy.stats as stats
import os

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

def plot_bar_with_error(ax, Qrange, hist, chance_means, chance_errors, title, xlabel, ylabel, colors_list):
    """Helper function to plot bar chart with error shading and chance means as horizontal bars."""
    perc = hist / np.sum(hist)
    ax.bar(Qrange, hist, width=0.9, color=colors_list, alpha=0.7)  # Added alpha for transparency

    # Update text positioning and formatting
    for x, y, p in zip(Qrange, hist, perc):
        ax.text(x - 0.4, 0, str(y) + '\n' + str(round(p * 100, 1)) + '%', ha='left', va='bottom', fontsize=9)  # Positioning text inside the canvas

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if chance_means is not None and chance_errors is not None:
        for q in Qrange:
            # Draw horizontal line for chance mean
            ax.hlines(y=chance_means[q], xmin=q - 0.4, xmax=q + 0.4, color='black', linewidth=2, label='Chance Mean' if q == 0 else "")
            # Draw error bar as horizontal line above the chance mean
            ax.hlines(y=chance_means[q] - chance_errors[q], xmin=q - 0.4, xmax=q + 0.4, color='black', linestyle='--', linewidth=1)
            ax.hlines(y=chance_means[q] + chance_errors[q], xmin=q - 0.4, xmax=q + 0.4, color='black', linestyle='--', linewidth=1)

    plt.xticks(Qrange)


def plot_selective_cells_histo(pref_num, n_numerosities, colors_list, excitatory_or_inhibitory=None, chance_means=None, chance_errors=None, file_name=None):
    """Main function to plot selective cells histogram with simplified save path."""
    Qrange = np.arange(n_numerosities)

    if excitatory_or_inhibitory is None:
        # Case for all neurons
        hist = [np.sum(pref_num == q) for q in Qrange]
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_bar_with_error(ax, Qrange, hist, chance_means, chance_errors, "All Neurons", 'Preferred Numerosity', 'Number of cells', colors_list)
        plt.xticks(Qrange)
        plt.legend()
    else:
        # Case with excitatory and inhibitory neurons
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Excitatory neurons
        excitatory_indices = excitatory_or_inhibitory == 'excitatory'
        hist_exc = [np.sum(pref_num[excitatory_indices] == q) for q in Qrange]
        plot_bar_with_error(axes[0], Qrange, hist_exc, 
                            chance_means['excitatory'], 
                            chance_errors['excitatory'], 
                            'Excitatory Neurons', 'Preferred Numerosity', 'Number of cells', colors_list)

        # Inhibitory neurons
        inhibitory_indices = excitatory_or_inhibitory == 'inhibitory'
        hist_inh = [np.sum(pref_num[inhibitory_indices] == q) for q in Qrange]
        plot_bar_with_error(axes[1], Qrange, hist_inh, 
                            chance_means['inhibitory'], 
                            chance_errors['inhibitory'], 
                            'Inhibitory Neurons', 'Preferred Numerosity', 'Number of cells', colors_list)

        # All neurons
        hist = [np.sum(pref_num == q) for q in Qrange]
        plot_bar_with_error(axes[2], Qrange, hist, 
                            chance_means['total'], 
                            chance_errors['total'], 
                            'All Neurons', 'Preferred Numerosity', 'Number of cells', colors_list)

        plt.tight_layout()

    # Save figure
    if file_name:
        try:
            if file_name.endswith('.svg'):
                plt.savefig(file_name)
            elif file_name.endswith('.png'):
                plt.savefig(file_name, dpi=900)
            else:
                print("Error: file_name should end with either '.svg' or '.png'")
                return
            #print(f"File saved successfully at: {os.path.abspath(file_name)}")
        except Exception as e:
            print(f"Error while saving file: {e}")

    plt.show()


def plot_tuning_curves(tuning_mat_exc, tuning_err_exc,  colors=None, tuning_mat_inh=None, tuning_err_inh=None, excitatory_or_inhibitory=None, save_name=None):
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

        # Save figure if save_name is provided
        # Save figure
        if save_name:
            try:
                if save_name.endswith('.svg'):
                    plt.savefig(save_name)
                elif save_name.endswith('.png'):
                    plt.savefig(save_name, dpi=900)
                else:
                    print("Error: save_name should end with either '.svg' or '.png'")
                    return
                #print(f"File saved successfully at: {os.path.abspath(save_name)}")
            except Exception as e:
                print(f"Error while saving file: {e}")

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
        
        if save_name:
            try:
                if save_name.endswith('.svg'):
                    plt.savefig(save_name)
                elif save_name.endswith('.png'):
                    plt.savefig(save_name, dpi=900)
                else:
                    print("Error: save_name should end with either '.svg' or '.png'")
                    return
                #print(f"File saved successfully at: {os.path.abspath(save_name)}")
            except Exception as e:
                print(f"Error while saving file: {e}")

        plt.show()
    
def populate_tuning_dicts(tuning_mat_exc, tuning_mat_inh, n_numerosities):
    distRange_abs_0 = np.arange(-(n_numerosities-1), n_numerosities).tolist()
    distRange_abs_1 = np.arange(n_numerosities).tolist()

    dist_tuning_dict_abs_0_exc = {str(i): [] for i in distRange_abs_0}
    dist_tuning_dict_abs_1_exc = {str(i): [] for i in distRange_abs_1}
    
    dist_tuning_dict_abs_0_inh = {str(i): [] for i in distRange_abs_0} if tuning_mat_inh is not None else None
    dist_tuning_dict_abs_1_inh = {str(i): [] for i in distRange_abs_1} if tuning_mat_inh is not None else None

    for pref_n in np.arange(n_numerosities):
        for n in np.arange(n_numerosities):
            dist_tuning_dict_abs_0_exc[str(n - pref_n)].append(tuning_mat_exc[pref_n][n])
            dist_tuning_dict_abs_1_exc[str(abs(n - pref_n))].append(tuning_mat_exc[pref_n][n])
            if tuning_mat_inh is not None:
                dist_tuning_dict_abs_0_inh[str(n - pref_n)].append(tuning_mat_inh[pref_n][n])
                dist_tuning_dict_abs_1_inh[str(abs(n - pref_n))].append(tuning_mat_inh[pref_n][n])

    return (distRange_abs_0, distRange_abs_1, dist_tuning_dict_abs_0_exc, dist_tuning_dict_abs_1_exc, dist_tuning_dict_abs_0_inh, dist_tuning_dict_abs_1_inh)

def calculate_avg_and_err(dist_tuning_dict):
    avg_tuning = [np.mean(dist_tuning_dict[key]) if dist_tuning_dict[key] else 0 for key in dist_tuning_dict.keys()]
    err_tuning = [np.nanstd(dist_tuning_dict[key]) / sqrt(len(dist_tuning_dict[key])) if len(dist_tuning_dict[key]) > 1 else 0 for key in dist_tuning_dict.keys()]
    return avg_tuning, err_tuning

def plot_avg_tunings(distRange_abs_0, distRange_abs_1, dist_avg_tuning_abs_0_exc, dist_err_tuning_abs_0_exc, dist_avg_tuning_abs_1_exc, dist_err_tuning_abs_1_exc, dist_avg_tuning_abs_0_inh, dist_err_tuning_abs_0_inh, dist_avg_tuning_abs_1_inh, dist_err_tuning_abs_1_inh, save_file=None):
    num_plots = 2 if dist_avg_tuning_abs_0_inh is None else 4
    fig, axs = plt.subplots(2, 2, figsize=(10, 8)) if dist_avg_tuning_abs_0_inh is not None else plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten() if dist_avg_tuning_abs_0_inh is not None else axs
    
    # Plot excitatory
    axs[0].errorbar(distRange_abs_0, dist_avg_tuning_abs_0_exc, yerr=dist_err_tuning_abs_0_exc, color='black')
    axs[0].set_title('Excitatory Neurons: Numerical Distance')
    
    axs[1].errorbar(distRange_abs_1, dist_avg_tuning_abs_1_exc, yerr=dist_err_tuning_abs_1_exc, color='black')
    axs[1].set_title('Excitatory Neurons: Absolute Distance')
    
    # Plot inhibitory if present
    if dist_avg_tuning_abs_0_inh is not None:
        axs[2].errorbar(distRange_abs_0, dist_avg_tuning_abs_0_inh, yerr=dist_err_tuning_abs_0_inh, color='black')
        axs[2].set_title('Inhibitory Neurons: Numerical Distance')

        axs[3].errorbar(distRange_abs_1, dist_avg_tuning_abs_1_inh, yerr=dist_err_tuning_abs_1_inh, color='black')
        axs[3].set_title('Inhibitory Neurons: Absolute Distance')
    
    plt.tight_layout()
    if save_file:
        plt.savefig(f'{save_file}.png')
    plt.show()

def t_test_to_dataframe(dist_tuning_dict_abs_0, dist_tuning_dict_abs_1, n_numerosities, file_name=None):
    distance_comparisons = [(i, i + 1) for i in range(-n_numerosities + 1, n_numerosities - 1)]
    
    data = {
        'Distance Pair': [],
        't-stat (d0)': [],
        'p-value (d0)': [],
        'df (d0)': [],
        't-stat (d1)': [],
        'p-value (d1)': [],
        'df (d1)': []
    }
    
    for d1, d2 in distance_comparisons:
        # For numerical distance (d0)
        if str(d1) in dist_tuning_dict_abs_0 and str(d2) in dist_tuning_dict_abs_0:
            t_stat, p_value = stats.ttest_ind(a=dist_tuning_dict_abs_0[str(d1)], b=dist_tuning_dict_abs_0[str(d2)], equal_var=False)
            df = len(dist_tuning_dict_abs_0[str(d1)]) + len(dist_tuning_dict_abs_0[str(d2)]) - 2
            data['Distance Pair'].append(f'{d1} vs {d2}')
            data['t-stat (d0)'].append(t_stat)
            data['p-value (d0)'].append(p_value)
            data['df (d0)'].append(df)
        else:
            data['Distance Pair'].append(f'{d1} vs {d2}')
            data['t-stat (d0)'].append(None)
            data['p-value (d0)'].append(None)
            data['df (d0)'].append(None)

        # For absolute distance (d1), only calculate for positive comparisons
        if d1 >= 0:  # Only consider non-negative comparisons for d1
            if str(abs(d1)) in dist_tuning_dict_abs_1 and str(abs(d2)) in dist_tuning_dict_abs_1:
                t_stat, p_value = stats.ttest_ind(a=dist_tuning_dict_abs_1[str(abs(d1))], b=dist_tuning_dict_abs_1[str(abs(d2))], equal_var=False)
                df = len(dist_tuning_dict_abs_1[str(abs(d1))]) + len(dist_tuning_dict_abs_1[str(abs(d2))]) - 2
                data['t-stat (d1)'].append(t_stat)
                data['p-value (d1)'].append(p_value)
                data['df (d1)'].append(df)
            else:
                data['t-stat (d1)'].append(None)
                data['p-value (d1)'].append(None)
                data['df (d1)'].append(None)
        else:
            # Fill with None for negative comparisons in d1
            data['t-stat (d1)'].append(None)
            data['p-value (d1)'].append(None)
            data['df (d1)'].append(None)
    
    df = pd.DataFrame(data)
    
    # Save to Excel or CSV if filename is provided
    if file_name:
        try:
            if file_name.endswith('.xlsx'):
                df.to_excel(file_name, index=False)
            elif file_name.endswith('.csv'):
                df.to_csv(file_name, index=False)
            else:
                print("Error: file_name should end with either '.xlsx' or '.csv'")
                return
            #print(f"File saved successfully at: {os.path.abspath(file_name)}")
        except Exception as e:
            print(f"Error while saving file: {e}")

    return df


def plot_abs_dist_tunings(tuning_mat_exc, n_numerosities, tuning_mat_inh=None, save_file=None, print_stats=True, plot_figures=True):
    # Step 1: Populate tuning dictionaries
    distRange_abs_0, distRange_abs_1, dist_tuning_dict_abs_0_exc, dist_tuning_dict_abs_1_exc, dist_tuning_dict_abs_0_inh, dist_tuning_dict_abs_1_inh = populate_tuning_dicts(tuning_mat_exc, tuning_mat_inh, n_numerosities)
    
    # Step 2: Calculate averages and errors for excitatory neurons
    dist_avg_tuning_abs_0_exc, dist_err_tuning_abs_0_exc = calculate_avg_and_err(dist_tuning_dict_abs_0_exc)
    dist_avg_tuning_abs_1_exc, dist_err_tuning_abs_1_exc = calculate_avg_and_err(dist_tuning_dict_abs_1_exc)
    
    # Step 3: Calculate averages and errors for inhibitory neurons if provided
    if tuning_mat_inh is not None:
        dist_avg_tuning_abs_0_inh, dist_err_tuning_abs_0_inh = calculate_avg_and_err(dist_tuning_dict_abs_0_inh)
        dist_avg_tuning_abs_1_inh, dist_err_tuning_abs_1_inh = calculate_avg_and_err(dist_tuning_dict_abs_1_inh)
    else:
        dist_avg_tuning_abs_0_inh = dist_avg_tuning_abs_1_inh = None
        dist_err_tuning_abs_0_inh = dist_err_tuning_abs_1_inh = None
    
    # Step 4: Plot figures (if plot_figures is True)
    if plot_figures:
        plot_avg_tunings(distRange_abs_0, distRange_abs_1, dist_avg_tuning_abs_0_exc, dist_err_tuning_abs_0_exc, dist_avg_tuning_abs_1_exc, dist_err_tuning_abs_1_exc, dist_avg_tuning_abs_0_inh, dist_err_tuning_abs_0_inh, dist_avg_tuning_abs_1_inh, dist_err_tuning_abs_1_inh, save_file)
    
    # Step 5: Print stats if enabled
    df_exc = t_test_to_dataframe(dist_tuning_dict_abs_0_exc, dist_tuning_dict_abs_1_exc, n_numerosities,'./anova_results/excitatory_average_tunings.csv')
    if print_stats:
        print("Excitatory Neurons t-tests:")
        print(df_exc)
    if tuning_mat_inh is not None:
        df_inh = t_test_to_dataframe(dist_tuning_dict_abs_0_inh, dist_tuning_dict_abs_1_inh, n_numerosities, './anova_results/inhibitory_average_tunings.csv')
        if print_stats:
            print("\nInhibitory Neurons t-tests:")
            print(df_inh)
    
    # Step 6: Return results
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

def replot_tuning_curves(output_real, output_shuffled, save_name=None):
    # Extract data from the real output dictionary
    exc_avg_tuning_abs_0_real = output_real['exc_avg_tuning_abs_0']
    exc_err_tuning_abs_0_real = output_real['exc_err_tuning_abs_0']
    exc_avg_tuning_abs_1_real = output_real['exc_avg_tuning_abs_1']
    exc_err_tuning_abs_1_real = output_real['exc_err_tuning_abs_1']
    inh_avg_tuning_abs_0_real = output_real['inh_avg_tuning_abs_0']
    inh_err_tuning_abs_0_real = output_real['inh_err_tuning_abs_0']
    inh_avg_tuning_abs_1_real = output_real['inh_avg_tuning_abs_1']
    inh_err_tuning_abs_1_real = output_real['inh_err_tuning_abs_1']
    distRange_abs_0_real = output_real['distRange_abs_0']
    distRange_abs_1_real = output_real['distRange_abs_1']

    # Extract data from the shuffled output dictionary
    exc_avg_tuning_abs_0_shuffled = output_shuffled['exc_avg_tuning_abs_0']
    exc_err_tuning_abs_0_shuffled = output_shuffled['exc_err_tuning_abs_0']
    exc_avg_tuning_abs_1_shuffled = output_shuffled['exc_avg_tuning_abs_1']
    exc_err_tuning_abs_1_shuffled = output_shuffled['exc_err_tuning_abs_1']
    inh_avg_tuning_abs_0_shuffled = output_shuffled['inh_avg_tuning_abs_0']
    inh_err_tuning_abs_0_shuffled = output_shuffled['inh_err_tuning_abs_0']
    inh_avg_tuning_abs_1_shuffled = output_shuffled['inh_avg_tuning_abs_1']
    inh_err_tuning_abs_1_shuffled = output_shuffled['inh_err_tuning_abs_1']
    distRange_abs_0_shuffled = output_shuffled['distRange_abs_0']
    distRange_abs_1_shuffled = output_shuffled['distRange_abs_1']

    # Set up the figure with a number of subplots
    num_plots = 2  # Initialize number of plots for excitatory neurons
    if inh_avg_tuning_abs_0_real is not None:  # Check if inhibitory data is available
        num_plots += 2  # Update number of plots to include inhibitory neurons
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 layout
        axs = axs.flatten()  # Flatten for easier indexing
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1x2 layout

    # Plot for excitatory neurons - numerical distance = 0
    axs[0].errorbar(distRange_abs_0_real, exc_avg_tuning_abs_0_real, 
                    yerr=exc_err_tuning_abs_0_real, 
                    color='blue', label='Real Data', capsize=3)
    axs[0].errorbar(distRange_abs_0_shuffled, exc_avg_tuning_abs_0_shuffled, 
                    yerr=exc_err_tuning_abs_0_shuffled, 
                    color='cyan', label='Shuffled Data', capsize=3)
    axs[0].set_xticks(distRange_abs_0_real)
    axs[0].set_xlabel('Numerical Distance')
    axs[0].set_ylabel('Normalized Neural Activity')
    axs[0].set_title('Numerical Distance Tuning Curve (Excitatory)')
    axs[0].legend()

    # Plot for excitatory neurons - absolute distance = 1
    axs[1].errorbar(distRange_abs_1_real, exc_avg_tuning_abs_1_real, 
                    yerr=exc_err_tuning_abs_1_real, 
                    color='blue', label='Real Data', capsize=3)
    axs[1].errorbar(distRange_abs_1_shuffled, exc_avg_tuning_abs_1_shuffled, 
                    yerr=exc_err_tuning_abs_1_shuffled, 
                    color='cyan', label='Shuffled Data', capsize=3)
    axs[1].set_xticks(distRange_abs_1_real)
    axs[1].set_xlabel('Absolute Numerical Distance')
    axs[1].set_ylabel('Normalized Neural Activity')
    axs[1].set_title('Absolute Numerical Distance Tuning Curve (Excitatory)')
    axs[1].legend()

    # If inhibitory neurons are provided, create their plots
    if inh_avg_tuning_abs_0_real is not None:
        # Plot for inhibitory neurons - numerical distance = 0
        axs[2].errorbar(distRange_abs_0_real, inh_avg_tuning_abs_0_real, 
                        yerr=inh_err_tuning_abs_0_real, 
                        color='red', label='Real Data', capsize=3)
        axs[2].errorbar(distRange_abs_0_shuffled, inh_avg_tuning_abs_0_shuffled, 
                        yerr=inh_err_tuning_abs_0_shuffled, 
                        color='orange', label='Shuffled Data', capsize=3)
        axs[2].set_xticks(distRange_abs_0_real)
        axs[2].set_xlabel('Numerical Distance')
        axs[2].set_ylabel('Normalized Neural Activity')
        axs[2].set_title('Numerical Distance Tuning Curve (Inhibitory)')
        axs[2].legend()

        # Plot for inhibitory neurons - absolute distance = 1
        axs[3].errorbar(distRange_abs_1_real, inh_avg_tuning_abs_1_real, 
                        yerr=inh_err_tuning_abs_1_real, 
                        color='red', label='Real Data', capsize=3)
        axs[3].errorbar(distRange_abs_1_shuffled, inh_avg_tuning_abs_1_shuffled, 
                        yerr=inh_err_tuning_abs_1_shuffled, 
                        color='orange', label='Shuffled Data', capsize=3)
        axs[3].set_xticks(distRange_abs_1_real)
        axs[3].set_xlabel('Absolute Numerical Distance')
        axs[3].set_ylabel('Normalized Neural Activity')
        axs[3].set_title('Absolute Numerical Distance Tuning Curve (Inhibitory)')
        axs[3].legend()

    plt.tight_layout()

    # Save figure if save_name is provided
    # Save figure
    if save_name:
        try:
            if save_name.endswith('.svg'):
                plt.savefig(save_name)
            elif save_name.endswith('.png'):
                plt.savefig(save_name, dpi=900)
            else:
                print("Error: save_name should end with either '.svg' or '.png'")
                return
            #print(f"File saved successfully at: {os.path.abspath(save_name)}")
        except Exception as e:
            print(f"Error while saving file: {e}")

    plt.show()