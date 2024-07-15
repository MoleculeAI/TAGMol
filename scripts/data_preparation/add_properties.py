import argparse
import multiprocessing
import os
import pickle
import tqdm

from utils.evaluation.guide_props import get_props, smiles_to_mol
from utils.data import parse_sdf_file


def process(row, args):
    _, ligand_sdf_path, *_ = row
    ligand_sdf_abs_path = os.path.join(args.data_dir, ligand_sdf_path)
    try:
        ligand_smiles = parse_sdf_file(ligand_sdf_abs_path)['smiles']
    except:
        return tuple([None for _ in range(len(row) + 1)])
    ligand_mol = smiles_to_mol(ligand_smiles)
    ligand_props = get_props(ligand_mol)
    row = (*row, ligand_props)
    return row


def main(args):
    with open(args.src_index_path, "rb") as f:
        index = pickle.load(f)
    print(f"Source rows count: {len(index)}")

    num_errors = 0
    final_index = []

    with multiprocessing.Pool() as pool:
        for result in pool.starmap(process, [(row, args) for row in index]):
            if result[0] is None:
                num_errors += 1
            final_index.append(result)

    print(f"Number of errors: {num_errors}")
    print(f"Destination rows count: {len(final_index)}")

    with open(args.dest_index_path, "wb") as f:
        pickle.dump(final_index, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_index_path", "-src", type=str, default="./data/guide/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl")
    parser.add_argument("--dest_index_path", "-dest", type=str, default="./data/guide/crossdocked_v1.1_rmsd1.0_pocket10/index_props.pkl")
    parser.add_argument("--data_dir", "-d", type=str, default="./data/guide/crossdocked_v1.1_rmsd1.0_pocket10")
    args = parser.parse_args()
    main(args)
