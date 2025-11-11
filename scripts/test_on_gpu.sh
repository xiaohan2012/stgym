#! /bin/bash

time python run_rct.py \
       +exp=hpooling \
       design_space=graph_clf \
       resource=gpu-2 \
       sample_size=2 \
       design_space.train.max_epoch=2 \
       mlflow.experiment_name=test
