```markdown
# ReactionT5 Model

This folder contains a local copy of the **ReactionT5** codebase, sourced from the original open‑source repository (source information listed below).
> **Modification Notice**: Partial original source directories have been removed; several custom evaluation & data‑processing scripts have been added for our retrosynthesis pipeline. Core model logic in `models.py` / `utils.py` remains unchanged. This copy is integrated into our larger project for convenient invocation.

## Original Source
- **Repository**: [https://github.com/sagawatatsuya/ReactionT5](https://github.com/sagawatatsuya/ReactionT5)
- **License**: MIT — see `LICENSE.txt` within this folder.
- **Original README**: Please consult the upstream repository for background knowledge, official training workflow and citation information.

## Purpose
We adopt ReactionT5 as the **reactant‑prediction / retrosynthesis** module in our reinforcement‑learning‑based retrosynthesis pipeline.
This folder encapsulates the external model codebase, isolating third‑party implementation from our main project source code and facilitating version tracking of the dependency.
We deleted unused original subdirectories (`forward_reaction_prediction`, `study_reproduction`, `yield_prediction`) and only retain files relevant to forward‑and‑retro reaction prediction. Custom scripts for data preparation, fine‑tuning and evaluation are supplemented for our research.

## Directory Structure
Below is the actual directory tree of our local `ReactionT5/` folder. Key files & folders are annotated.
```
ReactionT5/
├── test/                                 # Test related files
├── reactiont5_forward_retro/             # Core official code for forward & retrosynthesis task (retained from upstream)
├── data/                                 # Input / output data directory
├── model/                                # **Directory for placing downloaded pretrained model checkpoints**
│   └── (Put your downloaded checkpoint files here)
├── task_forward/                         # Scripts & config for forward reaction task
├── finetune_forward_retro.py             # Custom script: fine‑tune ReactionT5 for forward‑retro joint task
├── eval_forward_reactiont5.py            # Custom script: evaluation for forward / retrosynthesis prediction
├── prepare_reactiont5_forward_data.py    # Custom script: dataset pre‑processing for ReactionT5
├── models.py                             # Original: ReactionT5 model architecture definition (unmodified)
├── utils.py                              # Original: general helper utilities (unmodified)
├── generation_utils.py                   # Original: text generation helper functions
├── LICENSE.txt                           # Original MIT license file
└── README.md                             # This documentation
```

> **Note**: Compared with the upstream repository, folders including `forward_reaction_prediction`, `study_reproduction`, `yield_prediction` are removed as they are irrelevant to our retrosynthesis task. Log files and local note files are project‑specific and not part of the official release. Refer to the original GitHub repo for the complete upstream file tree.

## Pretrained Weights
**Pretrained ReactionT5 model weights and datasets are available on the [Hugging Face Hub](https://huggingface.co/sagawa).**

Before running inference or fine‑tuning, you **must manually download the model checkpoint files** and place them under the `model/` folder shown above.

```bash
# Make sure your working directory is ReactionT5/
mkdir -p model
# Download checkpoint via wget / curl / huggingface_hub
# Example:
# wget -O model/pytorch_model.bin https://huggingface.co/sagawa/xxx/resolve/main/pytorch_model.bin
```

Exact checkpoint filenames are documented in the original repository.
> ⚠️ **Important**: This repository does **NOT** contain pretrained weights. You need to acquire weight files separately from Hugging Face Hub.

## Quick Start

1. **Environment setup**
The official dependency is defined in `requirements.yaml` from upstream. You may build conda environment following original instructions.

2. **Place pretrained weights**
Download checkpoints and save them inside `./model/`.

3. **Data preprocessing (custom)**
```bash
python prepare_reactiont5_forward_data.py
```

4. **Fine‑tuning (custom)**
```bash
python finetune_forward_retro.py
```

5. **Run evaluation (custom)**
```bash
python eval_forward_reactiont5.py
```

> For original usage of the official `reactiont5_forward_retro` module, please check the upstream repository documentation.

## Custom Modifications Summary
1. Removed unused upstream directories: `forward_reaction_prediction`, `study_reproduction`, `yield_prediction`.
2. Added self‑developed scripts:
    - `prepare_reactiont5_forward_data.py`: dataset preparation
    - `finetune_forward_retro.py`: fine‑tuning entry
    - `eval_forward_reactiont5.py`: prediction evaluation
3. Created empty `model/` folder for external pretrained checkpoints.
4. Local log and note files (`test1.log`, `test2.log`, `bash.txt`) are for internal debugging only.

> Core model source files (`models.py`, `utils.py`, generation utilities inside `reactiont5_forward_retro/`) keep identical to original release, no code alteration.

## Integration with Parent RL Retrosynthesis Project
- This folder serves as a local dependency module for our reinforcement‑learning retrosynthesis project.
- Main‑project scripts can invoke functions / scripts inside this directory by relative or absolute paths.
- Model weights are not redistributed; users must download pretrained weights independently.

## Acknowledgements
We thank the original authors for open‑sourcing ReactionT5. Please cite their publication when you utilize this model for your research.


```
