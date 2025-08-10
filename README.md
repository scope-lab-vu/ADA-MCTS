# Act as You Learn: Adaptive Decision-Making in Non-Stationary Markov Decision Processes

This repository contains the implementation for [Act as You Learn: Adaptive Decision-Making in Non-Stationary Markov Decision Processes](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1301.pdf) (AAMAS 2024). This paper presents a heuristic search algorithm called **Adaptive Monte Carlo Tree Search (ADA-MCTS)**, designed for decision-making in non-stationary environments.

**Abstract**:  
> Sequential decision-making in non-stationary environments is a common challenge in many real-world domains such as emergency response and autonomous driving. Existing methods tend to assume that updated dynamics of the environment are known or adopt pessimistic planning strategies. In this paper, we introduce ADA-MCTS, an adaptive online planning algorithm that enables an agent to "act as it learns"—learning the updated environmental dynamics during interaction and adjusting its decision-making accordingly. By disintegrating aleatoric and epistemic uncertainty, ADA-MCTS safely explores unknown regions of the state space and maximizes rewards in familiar regions. We demonstrate the superior adaptability and performance of ADA-MCTS compared to baseline methods across several benchmark problems.

## Installation
To set up the environment for this project, use the provided `requirements.txt` file and Python 3.8.17:

```bash
python3.8 -m venv ada_mcts_env
source ada_mcts_env/bin/activate  
pip install -r requirements.txt
```
Or conda:
```bash
conda create -n adamcts -c conda-forge python=3.8
conda activate adamcts
pip install -r requirements.txt
```

## Running the Algorithm
Once the environment is set up, you can run the entire pipeline by executing the run.sh script. This script collects experiences, initializes the MDP0 model, and performs the Act As You Learn algorithm.
To run the algorithm, use the following command:
```bash
chmod +x run.sh
./run.sh
```

## Acknowledgements
We acknowledge the implementation of Bayesian Neural Network from the [Hidden Parameter Markov Decision Processes](https://github.com/dtak/hip-mdp-public.git) repository.


## Citation
```
@inproceedings{10.5555/3635637.3662988,
author = {Luo, Baiting and Zhang, Yunuo and Dubey, Abhishek and Mukhopadhyay, Ayan},
title = {Act as You Learn: Adaptive Decision-Making in Non-Stationary Markov Decision Processes},
year = {2024},
isbn = {9798400704864},
publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
address = {Richland, SC},
booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems},
pages = {1301–1309},
numpages = {9},
keywords = {monte carlo tree search, non-stationary environments, online planning, sequential decision-making},
location = {Auckland, New Zealand},
series = {AAMAS '24}
}
```
