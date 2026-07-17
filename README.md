# Land-then-Transport

Official code release for:

**Land-then-transport: A Flow Matching-Based Generative Decoder for Wireless Image Transmission**

Jingwen Fu, Ming Xiao, Mikael Skoglund, and Dong In Kim

The paper is published in **IEEE Transactions on Wireless Communications**, vol. 25, pp. 19757–19772, 2026 ([DOI: 10.1109/TWC.2026.3710439](https://doi.org/10.1109/TWC.2026.3710439), IEEE document 11604027). A preprint is available as [arXiv:2601.07512](https://arxiv.org/abs/2601.07512).

## Overview

Land-then-Transport (LTT) treats a received wireless observation as a point on a continuous-time flow at a channel-dependent landing time. A conditional flow matching model then transports that observation toward the clean image distribution using a deterministic ODE decoder.

This public version supports the core AWGN workflow:

- AWGN channels;
- conditional flow matching training;
- PSNR, MS-SSIM, and LPIPS evaluation.

Rayleigh and MIMO research modules are retained in the source tree for reference, but they are not part of the currently supported public workflow. Model weights and large experiment outputs are not included in this release.

## Repository structure

- `train.py`: conditional flow matching training.
- `evaluate.py`: checkpoint evaluation for the supported AWGN workflow.
- `flow_matching/`: schedules, ODE solver, channel conversion, metrics, datasets, and UNet.
- `ltt/config.py`: shared command-line configuration.
- `ltt/checkpoint.py`: checkpoint saving and loading.

## Installation

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[metrics]"
```

Weights & Biases logging is optional:

```bash
pip install -e ".[metrics,tracking]"
```

W&B is disabled by default and is enabled only with `--use_wandb true`.

## Data

The data root is selected in this order:

1. `--data_root`;
2. the `LTT_DATA_ROOT` environment variable;
3. `./data`.

MNIST, Fashion-MNIST, and CIFAR-10 can be downloaded through torchvision. DIV2K must be downloaded separately and arranged as:

```text
data/
├── DIV2K_train_HR/
│   └── *.png
└── DIV2K_valid_HR/
    └── *.png
```

The datasets remain subject to their original licenses and terms.

## Training

```bash
python train.py \
  --dataset mnist \
  --data_root ./data \
  --output_dir ./outputs \
  --n_epochs 100 \
  --sigma_max 1.0 \
  --sigma_schedule sqrt
```

The public training path learns the AWGN-aligned flow field.

## Evaluation

No pretrained weights are distributed with this repository. Complete training first, then pass the generated checkpoint explicitly:

```bash
python evaluate.py \
  --dataset mnist \
  --data_root ./data \
  --checkpoint ./outputs/mnist/ckpt_e100-schedsqrt-chawgn-smax1.0000-lr1e-03.pth \
  --channel awgn \
  --test_snr_db_list "0,5,10,15" \
  --ode_steps 10
```

## Citation

Please cite the published article:

```bibtex
@article{fu2026land,
  title   = {Land-Then-Transport: A Flow Matching-Based Generative Decoder for Wireless Image Transmission},
  author  = {Fu, Jingwen and Xiao, Ming and Skoglund, Mikael and Kim, Dong In},
  journal = {IEEE Transactions on Wireless Communications},
  volume  = {25},
  pages   = {19757--19772},
  year    = {2026},
  doi     = {10.1109/TWC.2026.3710439}
}
```

## Acknowledgements

We gratefully acknowledge [Keishi Ishihara's flow-matching implementation](https://github.com/keishihara/flow-matching), on which this repository is based. The ODE solver also adapts components from [Meta's flow_matching](https://github.com/facebookresearch/flow_matching), and the UNet implementation derives from [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion).

Please see [NOTICE](NOTICE) for detailed attribution and third-party licensing information.

## License

The repository is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license in [LICENSE](LICENSE). Third-party components remain subject to their respective licenses.
