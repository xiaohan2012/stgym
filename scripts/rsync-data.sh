#! /bin/bash

rsync -avzh --exclude '*/processed/' data/ cyy:stgym/data/
