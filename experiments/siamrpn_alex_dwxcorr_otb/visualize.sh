ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

const=(0)
chk=(24)
for i in ${const[*]};
    do
    for j in ${chk[*]};
	    do
        CUDA_VISIBLE_DEVICES=0  python -u \
           ../../tools/test_template_w_tracker.py --cfg config_template.yaml \
           --dataset OTB100 \
          --dataset_dir /media/wattanapongsu/4T/dataset/TrackingTest \
           --snapshot /media/wattanapongsu/4T/models/pysot/siamrpn_alex_otb.pth \
           --model_name enc_5e2_512 --k $i --ks 2 --chk $j --z_size 512\
           --netG_pretrained  checkpoint${j}_m.pth  --search_attack\
           --video Basketball \
           --export_video \
            --saved_dir /media/wattanapongsu/4T/temp/save
     done
done
#test_template
#test_template_w_tracker
#visualize
#video
# Basketball, Biker, BlurBody, Bird1, BlurCar1, Bolt, CarScale, Dancer
# DragonBaby, David3

# person12_2
# ants1, crabs1

