# TAGMol: Target-Aware Gradient-guided Molecule Generation

[[Paper]](https://arxiv.org/abs/2406.01650)

![TAGMol Framework](https://arxiv.org/html/2406.01650v1/x1.png)

---

## Installation

### Dependency

The code has been tested in the following environment:

### Install via Conda and Pip
```bash
conda create -n tagmol python=3.8.17
conda activate tagmol
conda install pytorch=1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg=2.2.0 -c pyg
conda install rdkit=2022.03.2 openbabel=3.1.1 tensorboard=2.13.0 pyyaml=6.0 easydict=1.9 python-lmdb=1.4.1 -c conda-forge

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

### (Alternatively) Install with .yml file

```bash
conda env create -f environment.yml
```

**IMPORTANT NOTE:** You might have to do the following to append the path of the root working directory.
```bash
export PYTHONPATH=".":$PYTHONPATH
```

-----

## Data and Checkpoints
The resources can be found [here](https://drive.google.com/drive/folders/1INaXCjVZCOQ_awNeGl5Xpsde_8JfnJiy?usp=drive_link). The data are inside `data` directory, the backbone model is inside `pretrained_models` and the guide checkpoints are inside `logs`.


## Training
### Training Diffusion model from scratch
```bash
python scripts/train_diffusion.py configs/training.yml
```
### Training Guide model from scratch


#### BA

```bash
python scripts/train_dock_guide.py configs/training_dock_guide.yml
```

#### QED

```bash
python scripts/train_dock_guide.py configs/training_dock_guide_qed.yml
```

#### SA

```bash
python scripts/train_dock_guide.py configs/training_dock_guide_sa.yml
```

*NOTE: The outputs are saved in `logs/` by default.*


---

## Sampling
### Sampling for pockets in the testset

#### BackBone
```bash
python scripts/sample_diffusion.py configs/sampling.yml --data_id {i} # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.
```

We have a bash file that can run the inference for the entire test set in a loop.
```bash
bash scripts/batch_sample_diffusion.sh configs/sampling.yml backbone
```
The output will be stored in `experiments/backbone`.
The following variables: `BATCH_SIZE`, `NODE_ALL`, `NODE_THIS` and `START_IDX`, can be modified in the script file, if required.


#### BackBone + Gradient Guidance
```bash
python scripts/sample_multi_guided_diffusion.py [path-to-config.yml] --data_id {i} # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.
```

To run inference on all 100 targets in the test set:
```bash
bash scripts/batch_sample_multi_guided_diffusion.sh [path-to-config.yml] [output-dir-name]
```

The outputs are stored in `experiments_multi/[output-dir-name]`when run using the bash file. The config files are available in `configs/noise_guide_multi`.
- Single-objective guidance
    - BA:       `sampling_guided_ba_1.yml`
    - QED:      `sampling_guided_qed_1.yml`
    - SA:       `sampling_guided_sa_1.yml`
- Dual-objective guidance
    - QED + BA:     `sampling_guided_qed_0.5_ba_0.5.yml`
    - SA + BA:      `sampling_guided_sa_0.5_ba_0.5.yml`
    - QED + SA:     `sampling_guided_qed_0.5_sa_0.5.yml`
- Multi-objective guidance (our main model)
    - QED + SA + BA:    `sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml`


For example, to run the multi-objective setting (i.e., our model):
```bash
bash scripts/batch_sample_multi_guided_diffusion.sh configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34.yml qed_0.33_sa_0.33_ba_0.34
```

---
## Evaluation

### Evaluating Guide models
```bash
python scripts/eval_dock_guide.py --ckpt_path [path-to-checkpoint.pt]
```

### Evaluation from sampling results
```bash
python scripts/evaluate_diffusion.py {OUTPUT_DIR} --docking_mode vina_score --protein_root data/test_set
```
The docking mode can be chosen from {qvina, vina_score, vina_dock, none}

_NOTE: It will take some time to prepare pqdqt and pqr files when you run the evaluation code with vina_score/vina_dock docking mode for the first time._

---
## Results

<table>
<tbody><tr>
<td rowspan="2">Methods</td>
<td colspan="2">Vina Score (↓)</td>
<td colspan="2">Vina Min (↓)</td>
<td colspan="2">Vina Dock (↓)</td>
<td colspan="2">High Affinity (↑)</td>
<td colspan="2">QED (↑)</td>
<td colspan="2">SA (↑)</td>
<td colspan="2">Diversity (↑)</td>
<td rowspan="2">Hit Rate % (↑)</td>
</tr>
<tr>
<!-- <td></td> -->
<td>Avg.</td>
<td>Med.</td>
<td>Avg.</td>
<td>Med.</td>
<td>Avg.</td>
<td>Med.</td>
<td>Avg.</td>
<td>Med.</td>
<td>Avg.</td>
<td>Med.</td>
<td>Avg.</td>
<td>Med.</td>
<td>Avg.</td>
<td>Med.</td>
<!-- <td></td> -->
</tr>
<tr>
<td>Reference</td>
<td>-6.36</td>
<td>-6.46</td>
<td>-6.71</td>
<td>-6.49</td>
<td>-7.45</td>
<td>-7.26</td>
<td>-</td>
<td>-</td>
<td>0.48</td>
<td>0.47</td>
<td>0.73</td>
<td>0.74</td>
<td>-</td>
<td>-</td>
<td>21</td>
</tr>
<tr>
<td>liGAN</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-6.33</td>
<td>-6.20</td>
<td>21.1%</td>
<td>11.1%</td>
<td>0.39</td>
<td>0.39</td>
<td>0.59</td>
<td>0.57</td>
<td>0.66</td>
<td>0.67</td>
<td>13.2</td>
</tr>
<tr>
<td>AR</td>
<td>-5.75</td>
<td>-5.64</td>
<td>-6.18</td>
<td>-5.88</td>
<td>-6.75</td>
<td>-6.62</td>
<td>37.9%</td>
<td>31.0%</td>
<td>0.51</td>
<td>0.50</td>
<td>0.63</td>
<td>0.63</td>
<td>0.70</td>
<td>0.70</td>
<td>12.9</td>
</tr>
<tr>
<td>Pocket2Mol</td>
<td>-5.14</td>
<td>-4.70</td>
<td>-6.42</td>
<td>-5.82</td>
<td>-7.15</td>
<td>-6.79</td>
<td>48.4%</td>
<td>51.0%</td>
<td>0.56</td>
<td>0.57</td>
<td>0.74</td>
<td>0.75</td>
<td>0.69</td>
<td>0.71</td>
<td>24.3</td>
</tr>
<tr>
<td>TargetDiff</td>
<td>-5.47</td>
<td>-6.30</td>
<td>-6.64</td>
<td>-6.83</td>
<td>-7.80</td>
<td>-7.91</td>
<td>58.1%</td>
<td>59.1%</td>
<td>0.48</td>
<td>0.48</td>
<td>0.58</td>
<td>0.58</td>
<td>0.72</td>
<td>0.71</td>
<td>20.5</td>
</tr>
<tr>
<td>DecompDiff</td>
<td>-4.85</td>
<td>-6.03</td>
<td>-6.76</td>
<td>-7.09</td>
<td>-8.48</td>
<td>-8.50</td>
<td>
64.8%</td>
<td>
78.6%</td>
<td>0.44</td>
<td>0.41</td>
<td>0.59</td>
<td>0.59</td>
<td>0.63</td>
<td>0.62</td>
<td>24.9</td>
</tr>
<tr>
<td>TAGMol</td>
<td>-7.02</td>
<td>-7.77</td>
<td>-7.95</td>
<td>-8.07</td>
<td>-8.59</td>
<td>-8.69</td>
<td>
69.8%</td>
<td>
76.4%</td>
<td>0.55</td>
<td>0.56</td>
<td>0.56</td>
<td>0.56</td>
<td>0.69</td>
<td>0.70</td>
<td>27.7</td>
</tr>
</tbody></table>

Due to space constraints, we only share the `eval_results` folder generated from the evaluation script. It can be found in the [same link](https://drive.google.com/drive/folders/1INaXCjVZCOQ_awNeGl5Xpsde_8JfnJiy?usp=drive_link) as other resources, inside `results` directory.

---

## Citation

```
@article{dorna2024tagmol,
  title={TAGMol: Target-Aware Gradient-guided Molecule Generation},
  author={Vineeth Dorna and D. Subhalingam and Keshav Kolluru and Shreshth Tuli and Mrityunjay Singh and Saurabh Singal and N. M. Anoop Krishnan and Sayan Ranu},
  journal={arXiv preprint arXiv:2406.01650},
  year={2024}
}
```

---

## Acknowledgements

This codebase was build on top of [TargetDiff](https://github.com/guanjq/targetdiff)
