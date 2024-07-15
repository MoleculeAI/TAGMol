import argparse
import os
import glob
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from utils.misc_prop import get_eval_scores
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_guide_model import DockGuideNet3D


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

     # Logging
    logger = misc.get_logger('eval')
    logger.info(args)

    # Load config
    logger.info(f'Loading model from {args.ckpt_path}')
    ckpt_restore = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    config = ckpt_restore['config']
    logger.info(f'ckpt_config: {config}')

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
        # heavy_only=config.data.heavy_only
        index_path=config.data.index_path
    )
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = DockGuideNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt_restore['model'])
    model.eval()
    # print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')


    def validate(it):
        # fix time steps
        sum_loss, sum_n = 0, 0
        ytrue_arr, ypred_arr = [], []
        ytrue_buckets = [[] for _ in range(len(np.linspace(0, model.num_timesteps - 1, 10)))]
        ypred_buckets = [[] for _ in range(len(np.linspace(0, model.num_timesteps - 1, 10)))]
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for itr, t in enumerate(np.linspace(0, model.num_timesteps - 1, 10).astype(int)):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    loss, pred = model.get_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step,
                        dock=batch[config.train.get("target", "vina_dock")],        # TODO: show warning if target not in config
                        return_pred=True
                    )

                    sum_loss += float(loss) * batch_size
                    sum_n += batch_size
                    ypred_arr.append(pred.view(-1))
                    ytrue_arr.append(batch[config.train.get("target", "vina_dock")])
                    ypred_buckets[itr].append(pred.view(-1))
                    ytrue_buckets[itr].append(batch[config.train.get("target", "vina_dock")])

        avg_loss = sum_loss / sum_n
        ypred_arr = torch.cat(ypred_arr).cpu().numpy().astype(np.float64)
        ytrue_arr = torch.cat(ytrue_arr).cpu().numpy().astype(np.float64)
        rmse = get_eval_scores(ypred_arr, ytrue_arr, logger)
        for itr, t in enumerate(np.linspace(0, model.num_timesteps - 1, 10).astype(int)):
            ypred_buckets_it = torch.cat(ypred_buckets[itr]).cpu().numpy().astype(np.float64)
            ytrue_buckets_it = torch.cat(ytrue_buckets[itr]).cpu().numpy().astype(np.float64)
            rmse = get_eval_scores(ypred_buckets_it, ytrue_buckets_it, logger, prefix=f"T={t}")
        logger.info(
            '[Validate] Iter %05d | Loss %.6f' % (
                it, avg_loss 
            )
        )
        return avg_loss


    try:
        best_loss, best_iter = None, None
        
        val_loss = validate(0)
        
    except KeyboardInterrupt:
        logger.info('Terminating...')
