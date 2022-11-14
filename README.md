
## Resource requirements
We test with Nvidia RTX 2070s. This project was implemented under Pytorch 1.4.0 and CUDA 11.2.


## PySOT 
This project was forked from [pysot](https://github.com/STVIR/pysot) repository. If you need to reproduce training our model, you need to setup pysot repository first.

## Installation
create new environment from requirements.yaml
```
conda env create --file requirements.yml
conda activate dfa

python setup.py build_ext --inplace

```

## Download SiamRPN++ pretrained
 - We also suggest to download this pretrained from [original repository](https://drive.google.com/open?id=1Cx_oHu6o0gNeH7F9zZrgevfAGdyWC4D5). 
 - Change pretrained name as siamrpn_r50_otb_model.pth

## Download Our pretrained DFA model
 - We only support google site, access [here](https://drive.google.com/file/d/1a5-dK2qsLQHqzQl6BkzMag9G1BQUmsKH/view?usp=share_link). 
 
## Download dataset
 - We only share biker frames and json of [OTB100 dataset](https://drive.google.com/drive/folders/10H_DNcP-adPoYPqdx-PrmxRwcXlpnu6i?usp=sharing) . You can download full download from providing dataset source. 
 
 
## Testing
Beforehand testing this step, you need to setup the pretrained path, dataset path and save path.
```
cd DFA/experiments/siamrpn_r50_l234_dwxcorr_otb
bash test_dfa.sh
```

## Example results of our DFA model
You can access:
[1. DFA + SiamRPN++](https://drive.google.com/drive/folders/1aka675kSCt5FtSPubq_e3MQ9Ww7ctfXp?usp=sharing)
[2. DFA transferability](https://drive.google.com/drive/folders/1L_UvBPqVWTYM2Z5HQNVnaNWz3cFUgUKB?usp=sharing)

## License
PySOT is released under the [Apache 2.0 license](https://github.com/wattanapong/DFA/blob/main/LICENSE).
