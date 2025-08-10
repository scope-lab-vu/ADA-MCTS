#!/bin/bash
# Collect experiences
python -m data_collection.data_generation
# Initialize MDP0
python -m data_collection.train_model
# Perform Act As You Learn
python act_learn.py