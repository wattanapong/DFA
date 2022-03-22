ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

python ../../tools/remove_opt.py 	--root /media/wattanapongsu/4T/temp/save/checkpoint/$1 --checkpoint $2
