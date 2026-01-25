#! /bin/bash

rsync -avzh --compress-level 9 --progress --exclude '*/processed/' data/ cyy2:stgym/data/
