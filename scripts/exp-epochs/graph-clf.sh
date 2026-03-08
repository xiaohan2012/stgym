#!/bin/bash

time python run_rct.py \
       +exp=epochs \
       design_space=graph_clf \
       resource=gpu \
       sample_size=100 \
       mlflow.experiment_name=graph-clf-epochs-$(date +%m-%d-%Y)
