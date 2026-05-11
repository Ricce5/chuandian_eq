export CUBLAS_WORKSPACE_CONFIG=:4096:8
python main.py --model reg_mixer_attnpl_t --mode train
python main.py --model clf_mixer_attnpl_t --mode train
python main.py --model mixer_tpp --mode test --checkpoint  checkpoints/mixer_tpp_20260204-233222
python main.py --model classifier --mode train
python main.py --model clf_attnpl_t --mode train
python main.py --model classifier_se --mode train
python main.py --model classifier_stm --mode train
python main.py --model classifier_tm_s --mode train
python main.py --model clf_tm_attnpl --mode train
python main.py --model clf_tm_attnpl_t --mode test  --checkpoint ./checkpoints/clf_mixer_attnpl_t_20250908-202537
python main.py --model clf_mixer_attnpl_t --mode test --checkpoint checkpoints/clf_mixer_attnpl_t_20260126-180606
python main.py --model clf_rnn 
python main.py --model thp --mode train
python main.py --model rtpp --mode test --checkpoint ./checkpoints/rtpp_20250907-144043
python main.py --model nhpp --mode train --checkpoint ./checkpoints/nhpp_20260317-174553
python main.py --model etas --mode test --checkpoint checkpoints/etas_20251231-123806
python main.py --model etas_zhuang --mode test
python main.py --model btpp --mode train
python main.py --model mtpp --mode train
python main.py --model mhp --mode train
python main.py --model mixer_tpp --mode test --checkpoint checkpoints/mixer_tpp_20260205-110204
python main.py --model thp_deltat --mode train
python main.py --model clf_tm_cv_attnpl_t --mode train
python main.py --model reg_mixer_attnpl_t --mode test --checkpoint checkpoints/reg_mixer_attnpl_t_20250907-183220
python main.py --model lstm --mode test --checkpoint ./checkpoints/lstm_20250906-152053

diff ./checkpoints/mixer_tpp_20250821-125414/config.yaml  ./checkpoints/mixer_tpp_20250907-112659/config.yaml



  python main.py \
    --model lstm \
    --mode test \
    --checkpoint_dir tmp/lstm_optuna_profiles/run_20x2/runs/global/optuna_trials/trial_0011_global \
    --ckpt_select last




  python scripts/run_clf_mf_tf_grid.py \
    --mfs 4,4.5,4.5,5,5.5 \
    --seeds 0,1,2 \
    --exp_name clf_mf_tf_multi_seed_$(date +%Y%m%d-%H%M%S)


  python scripts/run_clf_mf_tf_grid.py \
    --tfs 10,20,30,60,90 \
    --mfs 4,4.5,4.5,5,5.5 \
    --seeds 0,1,2 \
    --max_parallel 3 \
    --gpu_ids 0 \
    --exp_name clf_grid_parallel_1

  python scripts/run_clf_mf_tf_grid.py --exp_config config/experiments/clf_mf_tf_grid.yaml     --exp_name clf_grid_parallel_0

  python scripts/summarize_clf_mf_tf_grid.py --exp_dir experiments/clf_grid_9 --best_metric auc
   python scripts/summarize_reg_mixer_attnpl_grid.py --exp_dir experiments/reg_grid_inter9_excl2_round --out_dir /tmp/reg_grid_inter9_summary_check --ckpt_select last

d

  python scripts/run_reg_mixer_attnpl_grid.py --exp_config config/experiments/reg_mixer_attnpl_grid_lr_off.yaml
  python scripts/run_reg_mixer_attnpl_grid.py --exp_config config/experiments/reg_mixer_attnpl_grid.yaml
  
python scripts/summarize_reg_mixer_attnpl_grid.py --exp_dir experiments/reg_grid_inter9_excl2_round --best_metric rmse --best_mode min --ckpt_select last

  python scripts/run_reg_mixer_attnpl_t_optuna_profiles.py --profiles auto --optuna_trials 20 --out_dir /root/autodl-tmp/em_eqf/experiments/reg_mixer_attnpl_t_optuna_profiles --run_name
  run_$(date +%Y%m%d-%H%M%S)




python scripts/run_reg_mixer_attnpl_t_optuna_profiles.py --profiles auto --optuna_trials 3  --optuna_n_jobs 4  --out_dir /root/autodl-tmp/em_eqf/experiments/reg_mixer_attnpl_t_optuna_profiles \
    --run_name optuna_20_$(date +%Y%m%d-%H%M%S)


 python main.py --model reg_mixer_attnpl_t --mode optuna --config config/reg_mixer_attnpl_t.yaml --checkpoint_dir experiments/reg_mixer_attnpl_t_optuna_profiles/run_$(date +
        %Y%m%d-%H%M%S)/base --optuna_profile base --optuna_trials 3 --optuna_trial_test_ckpt_selects best,last


python scripts/run_reg_mixer_attnpl_t_optuna_profiles.py --profiles auto --optuna_trials 30  --optuna_trial_test_ckpt_selects best,last --out_dir /root/autodl-tmp/em_eqf/experiments/reg_mixer_attnpl_t_optuna_profiles --run_name run_$(date +%Y%m%d-%H%M%S)



python scripts/backfill_optuna_trial_tests.py --run_root /root/autodl-tmp/em_eqf/experiments/reg_mixer_attnpl_t_optuna_profiles/run_20260511-001210 --top_k 5 --ckpt_selects
best,last --config config/reg_mixer_attnpl_t.yaml --model reg_mixer_attnpl_t


  find experiments/reg_mixer_attnpl_t_optuna_profiles/run_20260511-000926/runs/low_lr/optuna_trials/trial_0001_low_lr -maxdepth 1 \( -name 'run.log' -o -name 'checkpoint_interrupted*.pth' -o
  -name 'tensorboard' \) -exec rm -rf {} +


  python scripts/analyze_optuna_last_val_loss_groups.py --trials-dir /root/autodl-tmp/em_eqf/experiments/reg_mixer_attnpl_t_optuna_profiles/run_20260511-000926/runs/base/optuna_trials