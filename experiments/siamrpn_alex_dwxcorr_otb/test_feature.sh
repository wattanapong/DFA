ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

const=(1)
chk=(02)
for i in ${const[*]};
    do
    for j in ${chk[*]};
	    do
        CUDA_VISIBLE_DEVICES=0  python -u \
           ../../tools/test_feature.py --cfg config_feature.yaml \
           --dataset VOT2018 \
          --dataset_dir /media/wattanapongsu/4T/dataset/TrackingTest \
           --snapshot /media/wattanapongsu/4T/models/pysot/siamrpn_alex_otb.pth \
           --model_name feat_1e3_r50_128 --k $i  --chk $j --z_size 128\
           --netG_pretrained  checkpoint${j}.pth \
            --saved_dir /media/wattanapongsu/4T/temp/save
     done
done

#video
# Basketball, Biker, BlurBody, Bird1, BlurCar1, Bolt
# GOT-10k_Test_000001