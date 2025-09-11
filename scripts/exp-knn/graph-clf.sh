#! /bin/bash

time python run_rct.py \
       +exp=knn \
       design_space=graph_clf \
       resource=gpu-2 \
       sample_size=100 \
       design_space.train.max_epoch=200 \
       design_space.data_loader.graph_const=knn \
       mlflow.experiment_name=graph-clf-knn-sep-11-2025
