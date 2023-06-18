preprocessing of dataset :
1. run ```deam_preprocess.ipynb```
2. run ```create_dataset/run.py```
# -----------------------------------------------------------------------------------------------------------------------------
train :
```CUDA_VISIBLE_DEVICES=<GPU #> python train.py --conditioning <none or continuous_concat>```
# -----------------------------------------------------------------------------------------------------------------------------
train with restart_dir :
```CUDA_VISIBLE_DEVICES=<GPU #> python train.py --restart_dir <model name> --conditioning <none or continuous_concat>```

<model name> : e.g. 20230531-145111 in ./output/
# -----------------------------------------------------------------------------------------------------------------------------
generate (demo) :
```CUDA_VISIBLE_DEVICES=<GPU #> python generate.py --model_dir <model name> --conditioning continuous_concat```
