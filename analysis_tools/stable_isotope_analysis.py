import pandas as pd
import numpy as np
import os
import re
import math
import glob
import getpass
from matplotlib import pyplot as plt
from matplotlib import colormaps
import matplotlib.colors as cm
from warnings import simplefilter

import sys
sys.path.insert(0, 'global/homes/t/tharwood/repos/')
from metatlas.io import feature_tools as ft

from tqdm.notebook import tqdm
from IPython.display import display

np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None  # default='warn'
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

CONTROLLED_VOCAB = ['InjBL', 'InjBl', 'blank', 'Blank', 'QC', 'ISTD']

def _format_peak_heights(peak_heights: pd.DataFrame) -> pd.DataFrame:
    """Reformate the peak heights table for further manipulation.
    
    The data is transpoed to be more compound-centric, rather than file-centric. 
    Header is then reset to new compound columns.
    """
    
    peak_heights = peak_heights.transpose().reset_index()
    header = peak_heights.iloc[0]
    peak_heights = peak_heights[1:]
    peak_heights.columns = header
    peak_heights.iloc[:, 6:] = peak_heights.iloc[:, 6:].astype(float)
    
    return peak_heights

def read_peak_heights(project_directory: str, experiment: str, polarity: str, 
                      workflow_name: str, rt_alignment_number: int, analysis_number: int, user: str | None = None) -> pd.DataFrame:
    """Read peak heights csv file created from Metatlas Targeted workflow."""
    
    assert polarity == "positive" or polarity == "negative"
    
    if polarity == "positive":
        short_pol = "POS"
    if polarity == "negative":
        short_pol = "NEG"
    
    experiment_output_dir = "{}_{}_{}_{}".format(getpass.getuser(), workflow_name, rt_alignment_number, analysis_number)
    targeted_output_dir = "{}_{}".format(workflow_name, experiment)
    ema_output_dir = "EMA-{}".format(short_pol)
    data_sheets_dir = "{}_data_sheets".format(short_pol)
    peak_height_file = "{}_peak_height.tab".format(short_pol)
    
    output_dir = os.path.join(project_directory, experiment, experiment_output_dir, "Targeted", targeted_output_dir, ema_output_dir, data_sheets_dir)
    peak_heights_path = os.path.join(output_dir, peak_height_file)
    
    peak_heights = pd.read_csv(peak_heights_path, sep='\t')
    peak_heights = _format_peak_heights(peak_heights)
    
    return peak_heights


def read_compound_atlas(project_directory: str, experiment: str, polarity: str,
                        workflow_name: str, rt_alignment_number: int, analysis_number: int, user: str | None = None) -> pd.DataFrame:
    """Read compound atlas csv file created from Metatlas Targeted workflow"""
    
    assert polarity == "positive" or polarity == "negative"
    
    if polarity == "positive":
        short_pol = "POS"
    if polarity == "negative":
        short_pol = "NEG"
        
    if user is None:
        user = getpass.getuser()
    
    experiment_output_dir = "{}_{}_{}_{}".format(user, workflow_name, rt_alignment_number, analysis_number)
    targeted_output_dir = "{}_{}".format(workflow_name, experiment)
    ema_output_dir = "EMA-{}".format(short_pol)
    
    output_dir = os.path.join(project_directory, experiment, experiment_output_dir, "Targeted", targeted_output_dir, ema_output_dir)
    
    try:
        compound_atlas_path = [atlas_file for atlas_file in glob.glob(os.path.join(output_dir, "CompoundAtlas_*.csv"))][0]
    except:
        print("No atlas not found in: {}".format(output_dir))
    
    compound_atlas = pd.read_csv(compound_atlas_path, index_col=0)
    
    return compound_atlas

def detect_noise(compound_intensities: np.ndarray[float, ...], expected_isotopic_ratio=0.01, max_percent_difference=100) -> list[int, ...]:
    """Return indicies of intensity vector that differ from expected natural abundance patterns.
    
    Noise is determined by comparing the actual intensity to the expected intensity given the natural abundance of the isotope.
    The maximum percent difference allows for some signal variation.
    """
    
    m0_i = compound_intensities[0]
    
    noise_idxs = []
    for c13_count, intensity in enumerate(compound_intensities):
        if intensity > m0_i:
            noise_idxs.append(c13_count)
            continue
        
        isotopic_multiplier = expected_isotopic_ratio ** c13_count
        expected_intensity = m0_i * isotopic_multiplier
        
        if abs(expected_intensity - intensity) > (expected_intensity*max_percent_difference):
            noise_idxs.append(c13_count)
        
    return noise_idxs


def convert_name_to_metatlas(compound_name: str) -> str:
    """Perform and series of regex string sanitation steps for paritiy with Metatlas output."""
    
    compound_name = compound_name.split('///')[0]
    compound_name = re.sub(r'\.', 'p', compound_name)  # 2 or more in regexp
    compound_name = re.sub(r'[\[\]]', '', compound_name)
    compound_name = re.sub('[^A-Za-z0-9+-]+', '_', compound_name)
    compound_name = re.sub('i_[A-Za-z]+_i_', '', compound_name)
    if compound_name[0] in ['_', '-']:
        compound_name = compound_name[1:]
    if compound_name[-1] == '_':
        compound_name = compound_name[:-1]
    compound_name = re.sub('[^A-Za-z0-9]{2,}', '', compound_name) #2 or more in regexp
    
    return compound_name

def make_intensity_vector(df_row: pd.core.series.Series) -> np.ndarray[float, ...]:
    """Return an array of values given a row of a pandas DataFrame."""
    
    intensity_vector = [float(i) for i in df_row]
    intensity_vector = np.array(intensity_vector)
    
    return intensity_vector 

def get_lab_intensity_sum(intensity_vector: np.ndarray[float, ...]) -> float:
    """Get sum of intensity vector, excluding the unlabeled molecule."""
    
    intensity_sum_lab = intensity_vector[1:].sum()
    
    return intensity_sum_lab

def filter_compound_columns(compound_name: str, compound_adduct: str, compound_columns: list[str, ...]) -> list[str, ...]:
    """Filter DataFrame columns by compound name and adduct."""
    
    metatlas_compound_name = convert_name_to_metatlas(compound_name)
    
    compound_name_columns = [column for column in compound_columns if metatlas_compound_name == "_".join(column.split("_")[1:-4])]
    compound_name_columns = [column for column in compound_name_columns if compound_adduct in column.split("_")[-2]]
    
    if compound_adduct == "M":
        compound_name_columns = [column for column in compound_name_columns if "H" not in column.split("_")[-2]]
    if compound_adduct == "M+H" or compound_adduct == "M-H":
        compound_name_columns = [column for column in compound_name_columns if "M" in column.split("_")[-2][0]]
        compound_name_columns = [column for column in compound_name_columns if "H2O" not in column.split("_")[-2]]
        
    return compound_name_columns

def get_lab_and_unlab_vectors(peak_heights: pd.DataFrame, compound_name: str, compound_adduct: str, 
                            compound_columns: list[str, ...], short_groups_unlab: list[str, ...], short_groups_lab: list[str, ...]) -> dict[str, tuple[str, np.ndarray[float, ...]]]:
    """Get the intensity vectors for the top unlabeled and lab labeled files with the highest lab signal."""
    
    compound_name_columns = filter_compound_columns(compound_name, compound_adduct, compound_columns)
    
    # sort values by M0 intensity so the lowest lab intensity vector isn't empty
    peak_heights.sort_values(compound_name_columns[0], ascending=False, inplace=True)
    
    # make new column names for intensity vector and lab intensity sum
    intensity_vector_col = "{}_intensity_vector".format(compound_name)
    intensity_sum_lab_col = "{}_lab_intensity_sum".format(compound_name)
    
    # calculate intensity vector from compound columns and lab intensity sum
    peak_heights[intensity_vector_col] = peak_heights[compound_name_columns].apply(make_intensity_vector, axis=1)
    peak_heights[intensity_sum_lab_col] = peak_heights[intensity_vector_col].apply(get_lab_intensity_sum)
    
    # get the intensity vectors of the highest and lowest lab sum rows
    highest_unlab_entry = peak_heights[peak_heights['short groupname'].isin(short_groups_unlab)].sort_values(intensity_sum_lab_col, ascending=False)[['short groupname', intensity_vector_col]].iloc[0]
    highest_lab_entry = peak_heights[peak_heights['short groupname'].isin(short_groups_lab)].sort_values(intensity_sum_lab_col, ascending=False)[['short groupname', intensity_vector_col]].iloc[0]

    min_and_max_lab_entries = {'lab':(highest_lab_entry['short groupname'], highest_lab_entry[intensity_vector_col]),
                               'unlab':(highest_unlab_entry['short groupname'], highest_unlab_entry[intensity_vector_col])} 
    
    return min_and_max_lab_entries

def retrieve_file_paths(peak_heights: pd.DataFrame, short_groups: list[str, ...], experiment: str) -> list[str, ...]:
    """Retrieve file paths that correspond with the sample short groups of interest."""
    
    all_file_paths = glob.glob(os.path.join("/global/cfs/cdirs/metatlas/raw_data/*/", experiment, "*.h5"))
    filtered_files = peak_heights[peak_heights['short groupname'].isin(short_groups)]['file'].tolist()
    filtered_file_paths = [file_path for file_path in all_file_paths if os.path.basename(file_path) in filtered_files]
    
    return filtered_file_paths

def get_file_short_group(file_path: str) -> str:
    """Convert the file path back into the short group name."""
    
    file_basename = os.path.basename(file_path)
    
    pol = file_basename.split("_")[9]
    group = file_basename.split("_")[12]
    
    return "_".join([pol, group])

def get_hex_colors(colormap, n):
    
    color_chunks = np.array_split(colormaps[colormap].colors, n)
    
    hex_colors = []
    for color_chunk in color_chunks:
        mid_color_val = color_chunk[math.floor(len(color_chunk)/2)]
        hex_colors.append(cm.to_hex(mid_color_val))
        
    return hex_colors

def plot_eic(ax, ms1_data, lcmsrun, label, color):
    
    xy = ms1_data[(ms1_data['lcmsrun_observed'] == lcmsrun) & 
                  (ms1_data['label'] == label)][['rt', 'i']]
   
    x = xy['rt']
    y = xy['i']

    ax.plot(x, y, alpha=0.6, linewidth=0.8, color=color, label=label.split(" ")[-2])
    ax.set_yscale('log')
    
def plot_file_eics(ax, ms1_data, lcmsrun_list, label, color):
    
    for lcmsrun in lcmsrun_list:
        plot_eic(ax, ms1_data, lcmsrun, label, color)
    
def plot_file_per_compound_eics(ax, ms1_data, lcmsrun_list, compound_key, colormap):
    
    labels = set(ms1_data[(ms1_data['label'].str.contains(r"\b{} M".format(re.escape(compound_key[0].rstrip())), regex=True)) & (ms1_data['label'].str.contains(compound_key[1], regex=False))]['label'].tolist())
    
    if '2M' in compound_key[1]:
        labels = [label for label in labels if "2M" in label]
    else:
        labels = [label for label in labels if "2M" not in label]
    
    iso_count = [int(label.split(" ")[-2][1:]) for label in labels]
    sorted_labels = sorted(dict(zip(iso_count, labels)).items())
    sorted_labels = [label[1] for label in sorted_labels]
    
    colors = get_hex_colors(colormap, len(sorted_labels))
    
    for i, label in enumerate(sorted_labels):
        plot_file_eics(ax, ms1_data, lcmsrun_list, label, colors[i])
           
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    
def generate_compound_keys(compound_atlas: pd.DataFrame) -> list[tuple[str, str], ...]:
    """Generate compound keys (name and adduct) from Metatlas compound atlas"""
    
    compound_names = compound_atlas['label'].str[:-3].tolist()
    compound_names = [name.rstrip() for name in compound_names]
    
    compound_adducts = compound_atlas['adduct'].str[1:-2].tolist()
    compound_adducts = [adduct.rstrip() for adduct in compound_adducts]

    compound_keys = list(set(zip(compound_names, compound_adducts)))
    compound_keys.sort()
    
    return compound_keys

def get_unlab_and_lab_groups(short_group_pairs):
    short_groups_unlab = [pair[0] for pair in short_group_pairs]
    short_groups_lab = [pair[1] for pair in short_group_pairs]
    
    return short_groups_unlab, short_groups_lab

def collect_sample_ms1_data(compound_atlas, polarity, sample_files):
    
    compound_atlas['label'] = compound_atlas.apply(lambda x: "{} {}".format(x.label, x.adduct), axis=1)
    
    experiment_input = ft.setup_file_slicing_parameters(compound_atlas, sample_files, base_dir=os.getcwd(), 
                                                    ppm_tolerance=compound_atlas['mz_tolerance'][0], extra_time=0.5, polarity=polarity)

    ms1_data = []
    display("Collecting MS1 data from samples files:")
    for file_input in tqdm(experiment_input, unit='file'):

        data = ft.get_data(file_input, save_file=False, return_data=True)
        data['ms1_data']['lcmsrun_observed'] = file_input['lcmsrun']

        ms1_data.append(data['ms1_data'])

    ms1_data = pd.concat(ms1_data).reset_index(drop=True)
    
    return ms1_data

def get_output_path(project_directory, experiment):
    output_path = os.path.join(project_directory, experiment, "SIL_outputs")
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    return output_path

def export_noise_detection_plots(peak_heights: pd.DataFrame, ms1_data: pd.DataFrame, sample_files: list[str, ...], 
                                 short_groups_unlab, short_groups_lab, compound_keys, output_path, polarity):
    
    sample_files_unlab = [file for file in sample_files if get_file_short_group(file) in short_groups_unlab]
    sample_files_lab = [file for file in sample_files if get_file_short_group(file) in short_groups_lab]
    
    peak_heights = peak_heights[peak_heights['short groupname'].isin(short_groups_unlab + short_groups_lab)] 
    compound_columns = [column for column in peak_heights.columns if polarity in column]

    plot_path = os.path.join(output_path, '{}_noise_detection_plots'.format(polarity))

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    noise_detection_data = []
    display("Exporting noise detection plots:")
    for compound_name, compound_adduct in tqdm(compound_keys, unit="compound"):

        fig, ax = plt.subplots(2, 2, sharex='col', gridspec_kw={'width_ratios': [2, 1.5]})
        fig.suptitle('{} {} Noise Detection Plots'.format(compound_name, compound_adduct), fontsize=16)

        plot_file_per_compound_eics(ax[0, 0], ms1_data, sample_files_lab, (compound_name, compound_adduct), "tab20")
        ax[0, 0].set_title("lab Enriched Sample Signal")

        plot_file_per_compound_eics(ax[1, 0], ms1_data, sample_files_unlab, (compound_name, compound_adduct), "tab20")
        ax[1, 0].set_title("Unenriched Sample Signal")
        ax[1, 0].set_xlabel("Retention Time (Minutes)")

        entries_lab_and_unlab = get_lab_and_unlab_vectors(peak_heights, compound_name, compound_adduct, compound_columns, short_groups_unlab, short_groups_lab)

        x_labels = np.array(["M{}".format(i) for i in range(len(entries_lab_and_unlab['lab'][1]))])

        ax[0, 1].scatter(x_labels, entries_lab_and_unlab['lab'][1], alpha=0.8)
        ax[0, 1].set_yscale('log')
        ax[0, 1].set_title(entries_lab_and_unlab['lab'][0])

        noise_unlab_idxs = detect_noise(entries_lab_and_unlab['unlab'][1])

        ax[1, 1].scatter(x_labels, entries_lab_and_unlab['unlab'][1], alpha=0.8)
        ax[1, 1].scatter(x_labels[noise_unlab_idxs], entries_lab_and_unlab['unlab'][1][noise_unlab_idxs], c='r', label="Probable Noise", alpha=0.8)
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_title(entries_lab_and_unlab['unlab'][0])
        ax[1, 1].legend()

        ax[1, 1].set_xlabel("Number of lab Atoms")

        compound_plot_path = os.path.join(plot_path, '{}_{}_noise_detection_plot.png'.format(compound_name, compound_adduct))

        fig.set_size_inches(14, 8)
        fig.savefig(compound_plot_path)
        plt.close(fig)

        noise_detection_data.append({'compound_name':compound_name, 'compound_adduct':compound_adduct, 
                                     'plot_path':compound_plot_path, 'remove_entry':False, 'remove_m_signals':[], 'all_m_signals':list(x_labels)})
        
    return noise_detection_data

def filter_compound_peak_heights(gui_selection_data: pd.DataFrame, peak_heights: pd.DataFrame, 
                                 compound_name: str, compound_adduct: str, output_path: str, polarity: str) -> None:
    """Remove columns from peak heights table based on GUI selections for one compound key."""
    
    compound_columns = filter_compound_columns(compound_name, compound_adduct, peak_heights.columns)
    
    signals_to_remove = gui_selection_data[(gui_selection_data['compound_name'] == compound_name) &
                                           (gui_selection_data['compound_adduct'] == compound_adduct)]['remove_m_signals'].values[0]
    
    if 'Remove All' in eval(signals_to_remove):
        remove_cols = compound_columns
    else:
        col_pats = []
        for signal in eval(signals_to_remove):
            c13_num = signal.split("Remove ")[1]
            col_pats.append(c13_num)

        remove_cols = []
        for col in compound_columns:
            if col.split("_")[-4] in col_pats:
                remove_cols.append(col)
            
    if len(remove_cols) == 0:
        return
    
            
    peak_heights.drop(columns=remove_cols, inplace=True)
    
def filter_and_save_peak_heights(peak_heights: pd.DataFrame, compound_keys: list, output_path: str, polarity: str) -> None:
    """Filter peak heights using GUI selections and save new peak height data."""
    
    gui_selection_data_path = os.path.join(output_path, "{}_gui_selection_data.csv".format(polarity))
    gui_selection_data = pd.read_csv(gui_selection_data_path)
    
    for compound_name, compound_adduct in compound_keys:
        filter_compound_peak_heights(gui_selection_data, peak_heights, compound_name, compound_adduct, output_path, polarity)
        
    filtered_peak_heights_path = os.path.join(output_path, "{}_filtered_peak_heights.csv".format(polarity))
    peak_heights.to_csv(filtered_peak_heights_path)
    

def generate_outputs(project_directory: str,
                     experiment: str,
                     polarity: str,
                     workflow_name: str,
                     rt_alignment_number: int,
                     analysis_number: int,
                     short_group_pairs: list[tuple[str, str]],
                     user: str | None = None):
    """Generate noise detection plots and collect data.
    
    Output is used in the noise detection GUI.
    """
    
    output_path = get_output_path(project_directory, experiment)
    
    peak_heights = read_peak_heights(project_directory, experiment, polarity, workflow_name, rt_alignment_number, analysis_number, user=user)
    compound_atlas = read_compound_atlas(project_directory, experiment, polarity, workflow_name, rt_alignment_number, analysis_number, user=user)
    
    compound_keys = generate_compound_keys(compound_atlas)

    short_groups_unlab, short_groups_lab = get_unlab_and_lab_groups(short_group_pairs)
    sample_files = retrieve_file_paths(peak_heights, short_groups_unlab + short_groups_lab, experiment)
    
    ms1_data = collect_sample_ms1_data(compound_atlas, polarity, sample_files)
    compound_data = export_noise_detection_plots(peak_heights, ms1_data, sample_files, short_groups_unlab, short_groups_lab, compound_keys, output_path, polarity)

    return peak_heights, compound_data, compound_keys, output_path