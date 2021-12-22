# #!/bin/bash

weight_dir='/home/clz/3facedet/swinface/runs/train_small/exp/weights/'

epochs=(155 150 140 130 120 110 100 90 80 70 60 40 20 5)

for epoch in ${epochs[*]}
do
    weight_name=${weight_dir}swin_epoch_$epoch.pth
    echo $weight_name
    # test
    python test_widerface.py -m $weight_name
    # evaluation
    cd widerface_evaluate
    
    name=$(basename $epoch)
    
    echo $name >> results.txt
    
    python evaluation.py >> results.txt
    
    cd ..
    
done
