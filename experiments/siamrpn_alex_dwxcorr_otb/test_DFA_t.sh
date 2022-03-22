ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

#
const=(0)
chk=24
for i in ${const[*]};
    do
    for j in ${chk[*]};
	    do

        CUDA_VISIBLE_DEVICES=0  python -u \
           ../../tools/test_DFA.py --cfg config_DFA.yaml \
           --dataset GOT-10k \
          --dataset_dir  /4T/dataset/TrackingTest \
           --snapshot  /4T/models/pysot/siamrpn_alex_otb.pth \
           --model_name enc_5e2_512 --k $i --ks 0 --chk $j --z_size 512\
           --netG_pretrained  checkpoint${j}_m.pth\
           --skip_exist \
            --saved_dir /4T/temp/save
      done
done
#            --export_video --video Biker \
#test_template_w_tracker
#  --ks 1   --search_attack

# --export_video --video Biker  \
#test_template_w_tracker
#test_template
#video
# Basketball, Biker, BlurBody, Bird1, BlurCar1, Bolt, CarScale, Dancer
# DragonBaby, David3

# person12_2
# ants1

