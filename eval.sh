#!/bin/bash

weight_dir='/home/clz/model/swinface/weights/'

for i in $weight_dir/*
do

  # test
  python test_widerface.py -m $i
  # evaluation
  cd widerface_evaluate

  name=$(basename $i)

  echo $name >> results.txt

  python evaluation.py >> results.txt

  cd ..

done

