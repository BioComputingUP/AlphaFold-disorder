"""
module load python3
module load dssp/3.0.10

python3 alphafold_disorder.py -w 25 -th_rsa `awk 'BEGIN{for (i=0.55; i<0.80; i=i+0.05 ) printf("%.2f ", i)}'` -th_lddt `awk 'BEGIN{for (i=0.1; i<0.2; i=i+0.05 ) printf("%.2f ", i)}'` -i data/caid/ -o data/predictions/caid/AlphaFold.out


find data/predictions/caid -maxdepth 1 -type f -name "*.out" -printf "%f\n" | grep -v binding | while read line; do python3 map_disprot_id.py data/predictions/caid/$line data/predictions/caid/disorder/D_$line data/new_id_list_unp.txt; done
find data/predictions/caid -maxdepth 1 -type f -name "*.out" -printf "%f\n" | while read line; do python3 map_disprot_id.py data/predictions/caid/$line data/predictions/caid/binding/B_$line data/new_id_list_unp.txt; done





python3 alphafold_disorder.py -i data/caid/ -o data/predictions/caid.out

"""


from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB import DSSP
import numpy as np
import pandas as pd
import argparse
import logging.config
import sys
import csv
from pathlib import Path


def moving_average(x, w):
    # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    return np.convolve(x, np.ones(w), 'valid') / w


def process_pdb(pdb_file, pdb_name, dssp_path="mkdssp"):

    # Load the structure
    structure = PDBParser(QUIET=True).get_structure("", pdb_file)

    # Calculate DSSP
    dssp = DSSP(structure[0], pdb_file, dssp=dssp_path)  # WARNING Check the path of mkdssp
    dssp_dict = dict(dssp)

    # Parse b-factor (pLDDT) and DSSP
    df = []
    for i, residue in enumerate(structure.get_residues()):
        lddt = residue["CA"].get_bfactor() / 100.0
        rsa = float(dssp_dict.get((residue.get_full_id()[2], residue.id))[3])
        df.append((pdb_name, i + 1, seq1(residue.get_resname()), lddt, 1 - lddt, rsa))
    df = pd.DataFrame(df, columns=['name', 'pos', 'aa', 'lddt', 'disorder', 'rsa'])

    return df


def make_prediction(df, window_rsa=[25], thresholds_rsa=[0.581]):

    for w in window_rsa:
        # Smooth disorder score (moving average)
        column_rsa_window = 'rsa-{}'.format(w)
        half_w = int((w - 1) / 2)
        df[column_rsa_window] = moving_average(np.pad(df['rsa'], (half_w, half_w), 'reflect'), half_w * 2 + 1)

        # Transofrm scores above RSA threshold
        for th_rsa in thresholds_rsa:
            column_rsa_binding = 'bind-{}'.format(th_rsa)
            df[column_rsa_binding] = df[column_rsa_window].copy()
            df.loc[df[column_rsa_window] > th_rsa, column_rsa_binding] = df.loc[df[column_rsa_window] > th_rsa, 'lddt'] * (1 - th_rsa) + th_rsa

    return df


def parse_args():

    parent_parser = argparse.ArgumentParser(add_help=False)

    group = parent_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--in_pdb_dir', type=str, help="A folder of PDB input files")
    group.add_argument('-id', '--in_dssp', type=str, help="A TSV input file with RSA and pLDDT (checkpoint file)")

    parent_parser.add_argument('-o', type=str, required=True, help="Output file name. Automatically generate multiple files")

    parent_parser.add_argument('-f', type=str, choices=['tsv', 'caid'], help="Output format")

    parent_parser.add_argument("-w_rsa", nargs="*", type=int, default=[25], help="Apply a moving average over window on the RSA")
    parent_parser.add_argument("-th_rsa", nargs="*", type=float, default=[0.581], help="In binding prediction, filter positions with RSA values under threshold")

    parent_parser.add_argument("-dssp", type=str, default="mkdssp", help="Path to mkdssp (3.x)")

    parent_parser.add_argument("-ll", type=str, choices=["notset", "debug", "info", "warning", "error", "critical"], default="info", help="Log level")

    main_parser = argparse.ArgumentParser(parents=[parent_parser])

    return main_parser.parse_args()


if __name__ == "__main__":

    # parse command line arguments
    args = parse_args()
    fout_path = Path(args.o)

    # Set logger
    logging.basicConfig(format='%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
                        level=logging.getLevelName(args.ll.upper()), stream=sys.stdout)
    logging.getLogger('numexpr').setLevel(logging.WARNING)  # Remove numexpr warning

    if args.in_pdb_dir:
        # Generate DSSP output from PDB files
        data = pd.DataFrame()
        p = Path(args.in_pdb_dir)
        c = 0
        for file in p.iterdir():
            if file.stat().st_size > 0:  # and "P52799" in file.stem:  # 'P13693', "P52799", 'P0AE72', 'Q13148'
                logging.debug("Processing PDB {}".format(file))
                data = data.append(process_pdb(file, file.stem, dssp_path=args.dssp))
                c += 1
                # if c > 2:
                #     break
            else:
                logging.debug("Empty file {}".format(file))
        # Write a TSV file
        fout_name = "{}/{}_data.tsv".format(fout_path.parent, fout_path.stem)
        data.to_csv(fout_name, sep="\t", quoting=csv.QUOTE_NONE)
        logging.info("DSSP data written in {}".format(fout_name))
    elif args.in_dssp:
        # Start from checkpoint file
        data = pd.read_csv(args.in_dssp, sep="\t")
        logging.info("DSSP data read from {}".format(args.in_dssp))
    else:
        data = None

    # Calculate predictions
    pred = pd.DataFrame()
    for name, pdb_data in data.groupby('name'):
        pred = pred.append(make_prediction(pdb_data.copy(),
                                   window_rsa=args.w_rsa,
                                   thresholds_rsa=args.th_rsa))
    logging.info("Prediction calculated")

    # Write to file
    if args.f == 'tsv':
        fout_name = "{}/{}_pred.tsv".format(fout_path.parent, fout_path.stem)
        pred.to_csv(fout_name, sep="\t", quoting=csv.QUOTE_NONE, index=False)
        logging.info("Prediction written in {}".format(fout_path))
    elif args.f == 'caid':
        methods = set(pred.head()) - {'name', 'pos', 'aa', 'lddt'}
        for method in methods:
            with open("{}/{}_{}{}".format(fout_path.parent, fout_path.stem, method, fout_path.suffix), "w") as fout:
                for name, pdb_pred in pred.groupby('name'):
                    fout.write(">" + name + "\n" + (pdb_pred['pos'].astype(str) + "\t" + pdb_pred['aa'] + "\t" + pdb_pred[method].astype(str)).str.cat(sep="\n") + "\n")
        logging.info("CAID prediction files written in {}/".format(fout_path.parent))






