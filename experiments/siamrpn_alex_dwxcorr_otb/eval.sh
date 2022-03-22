ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

python -u ../../tools/eval.py 	\
  --tracker_path ./results \
  --dataset_path  /4T/dataset/TrackingTest \
	--dataset GOT-10k        \
	--num 1 		 \
		--tracker_prefix '*'  \
		--config config_DFA.yaml

#--vis