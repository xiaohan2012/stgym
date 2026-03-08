#!/bin/bash

time python run_rct.py \
       +exp=epochs \
       design_space=node_clf \
       resource=gpu \
       sample_size=100 \
       mlflow.experiment_name=node-clf-epochs-$(date +%m-%d-%Y)
