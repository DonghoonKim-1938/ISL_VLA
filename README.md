# ISL-VLA: A Framework for Vision-Language-Action Models in Robotics

This repository provides a comprehensive framework for developing and evaluating Vision-Language-Action (VLA) models for robotics. It includes implementations of various state-of-the-art policies, support for multiple robot platforms, and a suite of tools for training, evaluation, and data management.

## Features

- **State-of-the-Art Policies:**
  - Action-Chunking Transformer (ACT)
  - Diffusion Policy
  - Pi0
  - Pi0-Fast
  - TDMPC
  - VQ-BeT

- **Supported Robots:**
  - LeKiwi
  - Generic Manipulators
  - Mobile Manipulators
  - Stretch

- **Modular and Extensible:** The framework is designed to be easily extended with new policies, robots, and environments.

- **Comprehensive Tooling:** Includes scripts for:
  - Training and evaluation
  - Real-time control and data recording
  - Dataset management and conversion

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ISL_VLA.git
   cd ISL_VLA
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment (e.g., conda or venv).
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The `scripts` directory contains the main entry points for using the framework.

### Training

To train a policy, use the `train.py` script. You will need to provide a configuration file.
```bash
python scripts/train.py --config_path=configs/train.py
```

### Evaluation

To evaluate a trained policy, use the `eval.py` script.
```bash
python scripts/eval.py --policy.path=/path/to/your/policy
```

### Real-time Control and Recording

The framework supports real-time control of robots and recording of demonstration data.
- **Recording:** `scripts/record.py`
- **Real-time Evaluation:** `scripts/eval_real_time.py`

## Project Structure

```
.
├── common/                # Core modules
│   ├── datasets/          # Dataset loading, processing, and management
│   ├── envs/              # Environment wrappers and utilities
│   ├── optim/             # Optimizers and schedulers
│   ├── policies/          # Policy implementations (ACT, Diffusion, etc.)
│   ├── robot_devices/     # Drivers and interfaces for various robots
│   └── utils/             # Miscellaneous utilities
├── configs/               # Configuration files for different experiments
├── scripts/               # High-level scripts for training, evaluation, etc.
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## Configuration

The `configs` directory contains the configuration files for various tasks. These files are written in Python and allow for flexible and dynamic configuration. The `default.py` file contains the base configuration, which can be overridden by other configuration files.

- `train.py`: Configuration for training a policy.
- `eval.py`: Configuration for evaluating a policy.
- `record.py`: Configuration for recording data.

## Citation

If you use this framework in your research, please consider citing the original papers for the implemented policies and the following:

```bibtex
@misc{isl-vla,
  author = {Your Name},
  title = {ISL-VLA: A Framework for Vision-Language-Action Models in Robotics},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/ISL_VLA}},
}
```