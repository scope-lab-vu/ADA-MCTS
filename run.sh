#!/bin/bash
# Collect experiences
python data_collection/data_generation.py
# Initialize MDP0
python data_collection/train_model.py
# Perform Act As You Learn
python act_learn.py