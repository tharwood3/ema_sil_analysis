import re
import os
import pandas as pd

from .stable_isotope_analysis import read_compound_atlas, get_output_path

# Derived from NIST Atomic Data. 
# Defined as the difference between the common isotope (C12, N14) and uncommon stable isotope (C13, N15).
ISOTOPE_DIFF_MASSES = {'C': 1.00335483507, 'N': 0.997034}

def count_element_from_formula(formula: str, labeled_atom: str) -> int:
    """Count labeled element occurence in formula."""
    
    formula_sub = formula.split(labeled_atom)[1]
    
    if formula_sub[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        element_count = 1
    else:
        element_count = re.split(r'[a-zA-Z]', formula_sub)[0]
    
    return int(element_count)

def make_labeled_row(row: pd.core.series.Series, atom_i: int, isotope_diff: float, mz_tolerance: int) -> dict[str, str | float | int]:
    """Make labeled row for SIL atlas generation."""
    
    id_notes = None
    inchi_key = None
    ms1_notes = None
    ms2_notes = None
    identification_notes = None
    
    if atom_i == 0:
        inchi_key = row.inchi_key
        id_notes = row.id_notes
        ms1_notes = row.ms1_notes
        ms2_notes = row.ms2_notes
        identification_notes = row.identification_notes

    labeled_row = row[['formula', 'rt_min', 'rt_max', 'rt_peak', 'adduct', 'polarity']].to_dict()
    
    labeled_row['label'] = "{} M{}".format(row.label, atom_i)
    labeled_row['inchi_key'] = inchi_key
    labeled_row['id_notes'] = id_notes
    labeled_row['ms1_notes'] = ms1_notes
    labeled_row['ms2_notes'] = ms2_notes
    labeled_row['identification_notes'] = identification_notes
    labeled_row['mz'] = row.mz + (atom_i * isotope_diff)
    labeled_row['mz_tolerance'] = mz_tolerance
    
    return labeled_row

def make_labeled_atlas(unlabeled_atlas: pd.DataFrame, labeled_atom: str, mz_tolerance: int) -> pd.DataFrame:
    """Make atlas with all possible isotopologues for a given labeled atom and unlabeled EMA atlas."""
    
    isotope_diff = ISOTOPE_DIFF_MASSES[labeled_atom]

    labeled_atlas = []
    for idx, row in unlabeled_atlas.iterrows():
        
        if pd.isna(row.formula):
            continue

        if labeled_atom not in row.formula:
            continue

        atom_num = count_element_from_formula(row.formula, labeled_atom)

        for atom_i in range(atom_num+1):

            labeled_atlas.append(make_labeled_row(row, atom_i, isotope_diff, mz_tolerance))

    labeled_atlas = pd.DataFrame.from_dict(labeled_atlas)
    
    return labeled_atlas

def generate_atlas_file(project_directory: str, experiment: str, polarity: str,
                        workflow_name: str, rt_alignment_number: int, analysis_number: int,
                        labeled_atom: str, mz_tolerance: int, csv_atlas_file_name: str, user: str | None = None) -> str:
    """Generates and saves isotopically labeled version of a given compound atlas for labeled atom of interest."""
    
    assert labeled_atom in ISOTOPE_DIFF_MASSES.keys()
    
    compound_atlas = read_compound_atlas(project_directory, experiment, polarity, workflow_name, rt_alignment_number, analysis_number, user)
    labeled_atlas = make_labeled_atlas(compound_atlas, labeled_atom, mz_tolerance)
    
    output_path = get_output_path(project_directory, experiment)
    
    labeled_atlas_path = os.path.join(output_path, csv_atlas_file_name)
    labeled_atlas.to_csv(labeled_atlas_path)
    
    return labeled_atlas_path