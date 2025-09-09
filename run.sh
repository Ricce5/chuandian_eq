export CUBLAS_WORKSPACE_CONFIG=:4096:8
python main.py --model classifier --mode train
python main.py --model clf_attnpl_t --mode train
python main.py --model classifier_se --mode train
python main.py --model classifier_stm --mode train
python main.py --model classifier_tm_s --mode train
python main.py --model clf_tm_attnpl --mode train
python main.py --model clf_tm_attnpl_t --mode test  --checkpoint /root/autodl-tmp/chuandian_eq/checkpoints/clf_mixer_attnpl_t_20250908-202537
python main.py --model clf_mixer_attnpl_t --mode test --checkpoint checkpoints/clf_mixer_attnpl_t_20250902-112802
python main.py --model thp --mode train
python main.py --model rtpp --mode test --checkpoint /root/autodl-tmp/chuandian_eq/checkpoints/rtpp_20250907-144043
python main.py --model etas --mode train
python main.py --model btpp --mode train
python main.py --model mtpp --mode train
python main.py --model mhp --mode train
python main.py --model mixer_tpp --mode train  
python main.py --model thp_deltat --mode train
python main.py --model clf_tm_cv_attnpl_t --mode train
python main.py --model reg_mixer_attnpl_t --mode test --checkpoint /root/autodl-tmp/chuandian_eq/checkpoints/reg_mixer_attnpl_t_20250907-133839
python main.py --model lstm --mode test --checkpoint /root/autodl-tmp/chuandian_eq/checkpoints/lstm_20250906-152053

diff /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250821-125414/config.yaml  /root/autodl-tmp/chuandian_eq/checkpoints/mixer_tpp_20250907-112659/config.yaml









