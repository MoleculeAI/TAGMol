import argparse
import os
import shutil

import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_guided_diffusion import sample_guided_diffusion_ligand
from utils.data import PDBProtein
from utils import reconstruct
from rdkit import Chem

from scripts.property_prediction.inference import get_model as get_guide_model
from utils.transforms_prop import FeaturizeProteinAtom as GuideFeaturizeProteinAtom
from utils.transforms_prop import FollowerFeaturizeLigandAtom as GuideFeaturizeLigandAtom
from datasets.protein_ligand import KMAP

def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )

    return data

def sample_for_pocket_guided(config_path, pdb_path, device, batch_size, result_path):
    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(config_path)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # Load Guide Checkpoint
    guide_ckpt = torch.load(config.guide_model.checkpoint, map_location=device)
    logger.info(f"Guide Training Config: {guide_ckpt['config']}")

    # Guide Transforms
    guide_protein_featurizer = GuideFeaturizeProteinAtom()
    guide_ligand_featurizer = GuideFeaturizeLigandAtom(mode=ckpt['config']['data']['transform']['ligand_atom_mode'])


    # Guide model
    guide_model = get_guide_model(guide_ckpt['config'], guide_protein_featurizer.feature_dim, guide_ligand_featurizer.feature_dim)
    guide_model.load_state_dict(guide_ckpt['model'])
    guide_model = guide_model.to(device)
    guide_model.eval()
    for param in guide_model.parameters():
        param.requires_grad = False

    # Load pocket
    data = pdb_to_pocket_data(pdb_path)
    data = transform(data)

    all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, all_pred_pos0_traj, time_list = sample_guided_diffusion_ligand(
        model=model,
        guide_model=guide_model,
        data=data,
        num_samples=config.sample.num_samples,
        kind=config.sample.guide_kind,
        gradient_scale_cord=config.sample.gradient_scale_cord,
        gradient_scale_categ=config.sample.gradient_scale_categ,
        batch_size=batch_size, 
        device=device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms
    )

    result = {
        'data': data,
        'pred_ligand_pos': all_pred_pos,
        'pred_ligand_v': all_pred_v,
        'pred_ligand_pos_traj': all_pred_pos_traj,
        'pred_ligand_v_traj': all_pred_v_traj,
        'pred_ligand_v0_traj': all_pred_v0_traj,
        'pred_ligand_vt_traj': all_pred_vt_traj,
        'pred_pos0_traj': all_pred_pos0_traj
    }
    logger.info('Sample done!')

    # reconstruction
    gen_mols = []
    n_recon_success, n_complete = 0, 0
    for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='add_aromatic')
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            gen_mols.append(None)
            continue
        n_recon_success += 1

        if '.' in smiles:
            gen_mols.append(None)
            continue
        n_complete += 1
        gen_mols.append(mol)
    result['mols'] = gen_mols
    logger.info('Reconstruction done!')
    logger.info(f'n recon: {n_recon_success} n complete: {n_complete}')

    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'sample.pt'))
    mols_save_path = os.path.join(result_path, f'sdf')
    os.makedirs(mols_save_path, exist_ok=True)
    for idx, mol in enumerate(gen_mols):
        if mol is not None:
            sdf_writer = Chem.SDWriter(os.path.join(mols_save_path, f'{idx:03d}.sdf'))
            sdf_writer.write(mol)
            sdf_writer.close()
    logger.info(f'Results are saved in {result_path}')
    misc.close_logger(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--pdb_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--result_path', type=str, default='./outputs_guided_pdb')
    parser.add_argument('--num_samples', type=int)
    args = parser.parse_args()

    
    sample_for_pocket_guided(config_path=args.config,
                        pdb_path=args.pdb_path, 
                        device=args.device,
                        batch_size=args.batch_size,
                        result_path=args.result_path)

    # logger = misc.get_logger('evaluate')
    # device = args.device
    # # Load config
    # config = misc.load_config(args.config)
    # logger.info(config)
    # misc.seed_all(config.sample.seed)

    # # Load checkpoint
    # ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    # logger.info(f"Training Config: {ckpt['config']}")

    # # Transforms
    # protein_featurizer = trans.FeaturizeProteinAtom()
    # ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    # ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    # transform = Compose([
    #     protein_featurizer,
    # ])

    # # Load model
    # model = ScorePosNet3D(
    #     ckpt['config'].model,
    #     protein_atom_feature_dim=protein_featurizer.feature_dim,
    #     ligand_atom_feature_dim=ligand_featurizer.feature_dim
    # ).to(args.device)
    # model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    # model.eval()
    # for param in model.parameters():
    #     param.requires_grad = False
    # logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # # Load Guide Checkpoint
    # guide_ckpt = torch.load(config.guide_model.checkpoint, map_location=device)
    # logger.info(f"Guide Training Config: {guide_ckpt['config']}")

    # # Guide Transforms
    # guide_protein_featurizer = GuideFeaturizeProteinAtom()
    # guide_ligand_featurizer = GuideFeaturizeLigandAtom(mode=ckpt['config']['data']['transform']['ligand_atom_mode'])


    # # Guide model
    # guide_model = get_guide_model(guide_ckpt['config'], guide_protein_featurizer.feature_dim, guide_ligand_featurizer.feature_dim)
    # guide_model.load_state_dict(guide_ckpt['model'])
    # guide_model = guide_model.to(device)
    # guide_model.eval()
    # for param in guide_model.parameters():
    #     param.requires_grad = False

    # # Load pocket
    # data = pdb_to_pocket_data(args.pdb_path)
    # data = transform(data)
    # if args.num_samples:
    #     config.sample.num_samples = args.num_samples

    # all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_guided_diffusion_ligand(
    #     model=model,
    #     guide_model=guide_model,
    #     data=data,
    #     num_samples=config.sample.num_samples,
    #     kind=config.sample.guide_kind,
    #     gradient_scale_cord=config.sample.gradient_scale_cord,
    #     gradient_scale_categ=config.sample.gradient_scale_categ,
    #     batch_size=args.batch_size, 
    #     device=args.device,
    #     num_steps=config.sample.num_steps,
    #     pos_only=config.sample.pos_only,
    #     center_pos_mode=config.sample.center_pos_mode,
    #     sample_num_atoms=config.sample.sample_num_atoms
    # )
    # result = {
    #     'data': data,
    #     'pred_ligand_pos': all_pred_pos,
    #     'pred_ligand_v': all_pred_v,
    #     'pred_ligand_pos_traj': pred_pos_traj,
    #     'pred_ligand_v_traj': pred_v_traj
    # }
    # logger.info('Sample done!')

    # # reconstruction
    # gen_mols = []
    # n_recon_success, n_complete = 0, 0
    # for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
    #     pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='add_aromatic')
    #     try:
    #         pred_aromatic = trans.is_aromatic_from_index(pred_v, mode='add_aromatic')
    #         mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
    #         smiles = Chem.MolToSmiles(mol)
    #     except reconstruct.MolReconsError:
    #         gen_mols.append(None)
    #         continue
    #     n_recon_success += 1

    #     if '.' in smiles:
    #         gen_mols.append(None)
    #         continue
    #     n_complete += 1
    #     gen_mols.append(mol)
    # result['mols'] = gen_mols
    # logger.info('Reconstruction done!')
    # logger.info(f'n recon: {n_recon_success} n complete: {n_complete}')

    # result_path = args.result_path
    # os.makedirs(result_path, exist_ok=True)
    # shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    # torch.save(result, os.path.join(result_path, f'sample.pt'))
    # mols_save_path = os.path.join(result_path, f'sdf')
    # os.makedirs(mols_save_path, exist_ok=True)
    # for idx, mol in enumerate(gen_mols):
    #     if mol is not None:
    #         sdf_writer = Chem.SDWriter(os.path.join(mols_save_path, f'{idx:03d}.sdf'))
    #         sdf_writer.write(mol)
    #         sdf_writer.close()
    # logger.info(f'Results are saved in {result_path}')
