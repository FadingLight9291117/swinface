
weight_path=/home/clz/log/new_swin_epoch_5.pth

python test_widerface.py -m $weight_path
# evaluation
cd widerface_evaluate

name=$(basename $weight_path)

echo $name >> results.txt

python evaluation.py >> results.txt

cd ..