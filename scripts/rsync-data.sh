#! /bin/bash

rsync -avzh --progress --exclude '*/processed/' data/ cyy:stgym/data/
