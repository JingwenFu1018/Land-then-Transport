# Land-then-Transport

Official code release for:

**Land-then-transport: A Flow Matching-Based Generative Decoder for Wireless Image Transmission**

Jingwen Fu, Ming Xiao, Mikael Skoglund, and Dong In Kim

The paper has been accepted for publication in **IEEE Transactions on Wireless Communications**. The formal IEEE DOI, volume, issue, and page numbers are not available yet. The current manuscript is available as [arXiv:2601.07512](https://arxiv.org/abs/2601.07512).

## Overview

Land-then-Transport (LTT) treats a received wireless observation as a point on a continuous-time flow at a channel-dependent landing time. A conditional flow matching model then transports that observation toward the clean image distribution using a deterministic ODE decoder.

This public version contains the core implementation for:

- AWGN channels;
- Rayleigh fading with MMSE pre-equalization;
- correlated and uncorrelated 2×2 MIMO channels;
- conditional flow matching training;
- PSNR, MS-SSIM, and LPIPS evaluation.

Model weights and large experiment outputs are not included in this release.

## Repository structure

- `train.py`: conditional flow matching training.
- `evaluate.py`: checkpoint evaluation over AWGN, Rayleigh, or 2×2 MIMO channels.
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

The public training path learns the AWGN-aligned flow field reused by the supported channel models.

## Evaluation

Pass a locally trained checkpoint explicitly:

```bash
python evaluate.py \
  --dataset mnist \
  --data_root ./data \
  --checkpoint ./outputs/mnist/ckpt_best_e100-schedsqrt-chawgn-smax1.0000-lr1e-03.pth \
  --channel awgn \
  --test_snr_db_list "0,5,10,15" \
  --ode_steps 10
```

Set `--channel rayleigh` or `--channel mimo` for the other supported channels. For correlated MIMO, set `--mimo_corr_rho` to a value satisfying `|rho| < 1`.

## Citation

Until the final IEEE bibliographic record is available, please cite:

```bibtex
@article{fu2026land,
  title   = {Land-then-transport: A Flow Matching-Based Generative Decoder for Wireless Image Transmission},
  author  = {Fu, Jingwen and Xiao, Ming and Skoglund, Mikael and Kim, Dong In},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2026},
  note    = {Accepted for publication; arXiv:2601.07512},
  eprint  = {2601.07512},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## Acknowledgements

We gratefully acknowledge [Keishi Ishihara's flow-matching implementation](https://github.com/keishihara/flow-matching), on which this repository is based. The ODE solver also adapts components from [Meta's flow_matching](https://github.com/facebookresearch/flow_matching), and the UNet implementation derives from [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion).

Please see [NOTICE](NOTICE) for detailed attribution and third-party licensing information.

## License

The repository is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license in [LICENSE](LICENSE). Third-party components remain subject to their respective licenses.
