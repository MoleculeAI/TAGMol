from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.QED import qed

from utils.evaluation.sascorer import compute_sa_score

def obeys_brenk(molecule):
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)
    return catalog.GetFirstMatch(molecule) is None

def get_props(molecule):
    qed_score = qed(molecule)
    sa_score = compute_sa_score(molecule)
    brenk_pass = int(obeys_brenk(molecule))
    return {
        'qed': qed_score,
        'sa': sa_score,
        'brenk_pass': brenk_pass,
    }

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)
