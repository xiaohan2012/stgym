python run_rct.py \
       +exp=bn \
       design_space=graph_clf \
       resource=gpu-2 \
       sample_size=1 \
       design_space.train.max_epoch=5 \
       mlflow.experiment_name=test
