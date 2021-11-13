ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

const=(0)
chk=24
for i in ${const[*]};
    do
    for j in ${chk[*]};
	    do
        CUDA_VISIBLE_DEVICES=0  python -u \
           ../../tools/cam.py --cfg ../siamrpn_r50_l234_dwxcorr_otb/config_DFA.yaml \
           --dataset OTB100 \
          --dataset_dir /4T/dataset/TrackingTest \
           --snapshot /4T/models/pysot/siamrpn_r50_otb_model.pth \
           --model_name enc_5e2_512 --k $i --ks 2 --chk $j --z_size 512\
           --netG_pretrained  checkpoint${j}_m.pth\
            --video Bird1 --export_video  --search_attack \
            --saved_dir /4T/temp/save
     done
done
#
# --export_video --video Biker  \
#test_template_w_tracker
#test_template
#video
# Basketball, Biker, BlurBody, Bird1, BlurCar1, Bolt, CarScale, Dancer
# DragonBaby, David3

# person12_2
# ants1

