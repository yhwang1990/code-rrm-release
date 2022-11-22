# Code and Data for Paper "Improved Algorithm for Regret Ratio Minimization in Multi-Objective Submodular Maximization"

This repository contains the source code and data for the experiments of our paper "Improved Algorithm for Regret Ratio Minimization in Multi-Objective Submodular Maximization" published in AAAI 2023. It includes the Python implementation of the COORDINATE and POLYTOPE algorithms by (Soma and Yoshida, AAAI 2017), the RRMS and RRMS∗ algorithms by (Feng and Qian, AAAI 2021), and our HS-RRM algorithm for two RRM problems, namely multi-objective weighted maximum coverage and multi-objective data summarization. Our implementation of COORDINATE, POLYTOPE, RRMS, and RRMS∗ are adapted from the original version at <http://www.lamda.nju.edu.cn/qianc/code_rrms.html> with improved efficiency in the submodular maximization oracle.

## Datasets

This repository provides all datasets we use in the experiments and the scripts for pre-processing (`data-summarization/pre-processing.py` and `max-cover/calculate_weights.py`).

## Instructions

### Prerequisites

Please install Python 3.8+ and all packages in `requirements.txt` before running the experiments

### Usage

Folders:

- `data-summarization/` contains the code for multi-objective data summarization.

- `max-cover/` contains the code for multi-objective weighted maximum coverage.

- `plot/` contains the processed experimental results and the scripts to draw the figures in the paper.

Scripts for the experiments on multi-objective weighted maximum coverage:

`cd max-cover`
`python3 run_max_cover_dx.py`

where `x` denotes the number of objectives from 2 to 7.

Scripts for the experiments on multi-objective data summarization:

`cd data-summarization`
`python3 run_data_summarization_dx.py`

where `x` denotes the number of objectives from 2 to 7.

## Contact

Please contact [Yanhao Wang](mailto:yhwang@dase.ecnu.edu.cn) for any question on this repository.
