cd /data1/yangkaixing/VM-R/code/BPVMR

CUDA_VISIBLE_DEVICES=2 nohup python test_qb_norm.py --exp_name=Data2 --videos_dir=./data/Music-Dance --qbnorm_mode mode1 \
                --load_epoch -1 --use_beat --mode double_loss > ./result/Data2_test_m2v_QBNorm.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python test_qb_norm.py --exp_name=Data2 --videos_dir=./data/Music-Dance --qbnorm_mode mode1 \
                --load_epoch -1 --use_beat --mode double_loss --metric v2t > ./result/Data2_test_v2m_QBNorm.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python test.py --exp_name=Data2 --videos_dir=./data/Music-Dance \
                --load_epoch -1  --use_beat --mode double_loss --metric v2t > ./result/Data2_test_v2m.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python test.py --exp_name=Data2 --videos_dir=./data/Music-Dance \
                --load_epoch -1  --use_beat --mode double_loss > ./result/Data2_test_m2v.log 2>&1 &



CUDA_VISIBLE_DEVICES=2 nohup python test_qb_norm.py --exp_name=Data1 --videos_dir=./data/Music-Motion --qbnorm_mode mode1 \
                --load_epoch -1 --use_beat --mode double_loss > ./result/Data1_test_m2v_QBNorm.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python test_qb_norm.py --exp_name=Data1 --videos_dir=./data/Music-Motion --qbnorm_mode mode1 \
                --load_epoch -1 --use_beat --mode double_loss --metric v2t > ./result/Data1_test_v2m_QBNorm.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python test.py --exp_name=Data1 --videos_dir=./data/Music-Motion \
                --load_epoch -1  --use_beat --mode double_loss > ./result/Data1_test_m2v.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python test.py --exp_name=Data1 --videos_dir=./data/Music-Motion \
                --load_epoch -1  --use_beat --mode double_loss --metric v2t > ./result/Data1_test_v2m.log 2>&1 &
