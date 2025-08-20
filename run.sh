export CUBLAS_WORKSPACE_CONFIG=:4096:8
python main.py --model classifier --mode train
python main.py --model clf_attnpl_t --mode train
python main.py --model classifier_se --mode train
python main.py --model classifier_stm --mode train
python main.py --model classifier_tm_s --mode train
python main.py --model clf_tm_attnpl --mode train
python main.py --model clf_tm_attnpl_t --mode train
python main.py --model clf_mixer_attnpl_t --mode test --checkpoint /root/autodl-tmp/chuandian_eq/checkpoints/clf_mixer_attnpl_t_20250820-163418
python main.py --model thp --mode train
python main.py --model rtpp --mode train
python main.py --model btpp --mode train
python main.py --model mtpp --mode train
python main.py --model mhp --mode train
python main.py --model mixer_tpp --mode train
python main.py --model thp_deltat --mode train
python main.py --model clf_tm_cv_attnpl_t --mode train