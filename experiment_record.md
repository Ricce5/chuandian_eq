# Experiment Record

## 1. CLF

**配置字段：** `dataset | Twindow | Tfore | dt | Mf | context_len | p/n`

### 1.1 ChuanDian	180	10	10	4	1	1
    RF: /root/autodl-tmp/em_eqf/checkpoints/rf_6a2dcdd9
    Validation metrics:
```python
{'precision': 0.7412587412587412, 'recall': 0.9906542056074766, 'f1': 0.848, 'auc': 0.6380301941049604, 'pr_auc': 0.8205307899889852, 'fpr': 0.9487179487179487, 'tpr': 0.9906542056074766, 'R': 0.041936256889527956, 'conf': 9.583398310665321e-18, 'threshold': 0.23}
```

    Test metrics:
```python
{'precision': 0.5461538461538461, 'recall': 0.8987341772151899, 'f1': 0.6794258373205742, 'auc': 0.6508749069247952, 'pr_auc': 0.6893816052893005, 'fpr': 0.8676470588235294, 'tpr': 0.8987341772151899, 'R': 0.49084712755598825, 'conf': 2.4049131815839726e-22, 'threshold': 0.23, 'threshold_mode': 'val_threshold', 'val_threshold': 0.23}
```

    EM-EQF: checkpoints/clf_mixer_attnpl_t_20260205-222827
     val:
```python
{'precision': 0.7464788732394366, 'recall': 0.9906542056074766, 'f1': 0.8514056224899599, 'auc': 0.8140426551641505, 'pr_auc': 0.9271271534201313, 'fpr': 0.9230769230769231, 'tpr': 0.9906542056074766,
'R': 0.0675772825305535, 'conf': 1.8110423215777166e-11, 'threshold': 0.4134446680545807}
```
      test:
```python
{'precision': 0.6476190476190476, 'recall': 0.8607594936708861, 'f1': 0.7391304347826086, 'auc': 0.7961653015636635, 'pr_auc': 0.8651239527002411, 'fpr': 0.5441176470588235, 'tpr': 0.8607594936708861,
'R': 0.3166418466120626, 'conf': 0.9990411429982295, 'threshold': 0.44980189204216003}
```
```python
{'precision': 0.5384615384615384, 'recall': 0.9746835443037974, 'f1': 0.6936936936936937, 'auc': 0.7961653015636635, 'pr_auc': 0.8651239527002411, 'fpr': 0.9705882352941176, 'tpr': 0.9746835443037974, 'R': 0.004095309009679804, 'conf': 1.9126111789139523e-64, 'threshold': 0.4134446680545807}
```

### 1.2 ChuanDian	180	20	10	4.5	1	1
    RF: checkpoints/rf_052673bb
    Validation metrics:
```python
{'precision': 0.4411764705882353, 'recall': 1.0, 'f1': 0.6122448979591837, 'auc': 0.5689922480620155, 'pr_auc': 0.45413571962545823, 'fpr': 0.8837209302325582, 'tpr': 1.0, 'R': 0.11627906976744184, 'conf': 1.168744382534095e-33, 'threshold': 0.29}
```

    Test metrics:
```python
{'precision': 0.32407407407407407, 'recall': 1.0, 'f1': 0.48951048951048953, 'auc': 0.8415929203539824, 'pr_auc': 0.5902175941800013, 'fpr': 0.6460176991150443, 'tpr': 1.0, 'R': 0.32407407407407407, 'conf': 0.0001885330305711234, 'threshold': 0.29, 'threshold_mode': 'val_threshold', 'val_threshold': 0.29}
```


    EM-EQF: checkpoints/clf_mixer_attnpl_t_20260205-223841
    val:
```python
{'precision': 0.5145631067961165, 'recall': 0.8833333333333333, 'f1': 0.6503067484662577, 'auc': 0.7248062015503877, 'pr_auc': 0.6949490218519075, 'fpr': 0.5813953488372093, 'tpr': 0.8833333333333333,
'R': 0.30193798449612397, 'conf': 0.726927832278985, 'threshold': 0.3267044425010681}
```
    test:
```python
{'precision': 0.75, 'recall': 0.6, 'f1': 0.6666666666666666, 'auc': 0.8346396965865992, 'pr_auc': 0.7392454081253295, 'fpr': 0.061946902654867256, 'tpr': 0.6,
'R': 0.5380530973451327, 'conf': 1.0, 'threshold': 0.37116697430610657}
```
```python
{'precision': 0.379746835443038, 'recall': 0.8571428571428571, 'f1': 0.5263157894736842, 'auc': 0.8346396965865992, 'pr_auc': 0.7392454081253295, 'fpr': 0.4336283185840708, 'tpr': 0.8571428571428571,
'R': 0.4235145385587863, 'conf': 0.9999993605589341, 'threshold': 0.3267044425010681}
```

### 1.3 ChuanDian	180	30	10	4.5	1	1.5	~
    RF: checkpoints/rf_ba7d0dba
    Validation metrics:
```python
{'precision': 0.6102941176470589, 'recall': 1.0, 'f1': 0.7579908675799086, 'auc': 0.6092369477911647, 'pr_auc': 0.6999631182986612, 'fpr': 0.8833333333333333, 'tpr': 1.0, 'R': 0.1166666666666667, 'conf': 8.485617580824126e-16, 'threshold': 0.37}
```

    Test metrics:
```python
{'precision': 0.45098039215686275, 'recall': 0.9787234042553191, 'f1': 0.6174496644295302, 'auc': 0.8877437943262412, 'pr_auc': 0.7716930707750702, 'fpr': 0.5833333333333334, 'tpr': 0.9787234042553191, 'R': 0.44138506466416355, 'conf': 0.636473877611916, 'threshold': 0.37, 'threshold_mode': 'val_threshold', 'val_threshold': 0.37}
```

    EM-EQF: checkpoints/clf_mixer_attnpl_t_20260205-213430
    val:
```python
{'precision': 0.6377952755905512, 'recall': 0.9759036144578314, 'f1': 0.7714285714285715, 'auc': 0.7600401606425703, 'pr_auc': 0.8371695266694815, 'fpr': 0.7666666666666667, 'tpr': 0.9759036144578314,
'R': 0.20923694779116464, 'conf': 0.0019910004391618425, 'threshold': 0.2959364354610443}
```
    test:
```python
{'precision': 0.9090909090909091, 'recall': 0.6382978723404256, 'f1': 0.75, 'auc': 0.8993794326241135, 'pr_auc': 0.8516222215159499, 'fpr': 0.03125, 'tpr': 0.    6382978723404256,
'R': 0.6070478723404256, 'conf': 1.0, 'threshold': 0.47319287061691284}
```
```python
{'precision': 0.5287356321839081, 'recall': 0.9787234042553191, 'f1': 0.6865671641791045, 'auc': 0.8993794326241135, 'pr_auc': 0.8516222215159499, 'fpr': 0.4270833333333333, 'tpr': 0.9787234042553191, 'R': 0.5516400709219857, 'conf': 0.999999999986129, 'threshold': 0.2959364354610443}
```

### 1.4 ChuanDian	180	60	10	5	1	1	~*
    RF: checkpoints/rf_fd4476d8
        Validation metrics:
```python
{'precision': 0.7865168539325843, 'recall': 0.8045977011494253, 'f1': 0.7954545454545454, 'auc': 0.7631787554498614, 'pr_auc': 0.8223449072897118, 'fpr': 0.3275862068965517, 'tpr': 0.8045977011494253, 'R': 0.4770114942528736, 'conf': 1.0, 'threshold': 0.55}
```

    Test metrics:
```python
{'precision': 0.7647058823529411, 'recall': 0.5416666666666666, 'f1': 0.6341463414634146, 'auc': 0.8305976430976431, 'pr_auc': 0.7162431298335264, 'fpr': 0.08080808080808081, 'tpr': 0.5416666666666666, 'R': 0.4142156862745097, 'conf': 1.0, 'threshold': 0.55, 'threshold_mode': 'val_threshold', 'val_threshold': 0.55}
```

    EM-EQF: checkpoints/clf_mixer_attnpl_t_20260205-225249
    val:
```python
{'precision': 0.6434108527131783, 'recall': 0.9540229885057471, 'f1': 0.7685185185185185, 'auc': 0.7316686484344035, 'pr_auc': 0.8529366516296851, 'fpr': 0.7931034482758621, 'tpr': 0.9540229885057471,
'R': 0.16091954022988497, 'conf': 3.86480550837049e-05, 'threshold': 0.2606133818626404}
```
    test:
```python
{'precision': 0.6666666666666666, 'recall': 0.875, 'f1': 0.7567567567567568, 'auc': 0.8878367003367003, 'pr_auc': 0.849152712364616, 'fpr': 0.21212121212121213, 'tpr': 0.875,
'R': 0.6628787878787878, 'conf': 1.0, 'threshold': 0.2804109752178192}
```
```python
{'precision': 0.3609022556390977, 'recall': 1.0, 'f1': 0.5303867403314917, 'auc': 0.8878367003367003, 'pr_auc': 0.849152712364616, 'fpr': 0.8585858585858586, 'tpr': 1.0, 'R': 0.14141414141414144, 'conf': 1.2700833800545256e-35, 'threshold': 0.2606133818626404}
```


### 1.5 ChuanDian	180	90	10	5.5	1	0.5	~
     RF: checkpoints/rf_af684ff4
      Validation metrics:
```python
{'precision': 0.7297297297297297, 'recall': 0.8709677419354839, 'f1': 0.7941176470588235, 'auc': 0.9032258064516129, 'pr_auc': 0.8755510236689259, 'fpr': 0.2222222222222222, 'tpr': 0.8709677419354839, 'R': 0.6487455197132617, 'conf': 1.0, 'threshold': 0.33}
```

    Test metrics:
```python
{'precision': 0.19298245614035087, 'recall': 0.5789473684210527, 'f1': 0.2894736842105263, 'auc': 0.7504873294346979, 'pr_auc': 0.33728665089030824, 'fpr': 0.34074074074074073, 'tpr': 0.5789473684210527, 'R': 0.11172668513388735, 'conf': 0.9999999999999941, 'threshold': 0.33, 'threshold_mode': 'val_threshold', 'val_threshold': 0.33}
```

    EM-EQF: checkpoints/clf_mixer_attnpl_t_20260206-100920 checkpoints/clf_mixer_attnpl_t_20260205-230257
    val:
```python
{'precision': 0.926829268292683, 'recall': 0.6129032258064516, 'f1': 0.7378640776699029, 'auc': 0.8639784946236558, 'pr_auc': 0.8536683537619597, 'fpr': 0.03333333333333333, 'tpr': 0.6129032258064516,
'R': 0.5795698924731183, 'conf': 1.0, 'threshold': 0.4257252514362335}
```
    test:
```python
{'precision': 0.2191780821917808, 'recall': 0.8421052631578947, 'f1': 0.34782608695652173, 'auc': 0.7489278752436648, 'pr_auc': 0.27135010976409285, 'fpr': 0.4222222222222222, 'tpr': 0.8421052631578947,
'R': 0.41988304093567247, 'conf': 0.999998000327738, 'threshold': 0.3488326668739319}
```
```python
{'precision': 0.25, 'recall': 0.05263157894736842, 'f1': 0.08695652173913043, 'auc': 0.7489278752436648, 'pr_auc': 0.27135010976409285, 'fpr': 0.022222222222222223, 'tpr': 0.05263157894736842, 'R': 0.030409356725146195, 'conf': 1.0, 'threshold': 0.4257252514362335}
```



## 2. CLF Pre-training
None: checkpoints/clf_mixer_attnpl_t_20260205-220907
```python
{
"precision": 0.717391304347826,
"recall": 0.7021276595744681,
"f1": 0.7096774193548387,
"auc": 0.8353280141843972,
"pr_auc": 0.8148300015050373,
"fpr": 0.13541666666666666,
"tpr": 0.7021276595744681,
"R": 0.5667109929078015,
"conf": 1.0,
"threshold": 0.37369078397750854
}
```
AZDX-b：checkpoints/clf_mixer_attnpl_t_20260205-213729 checkpoints/clf_mixer_attnpl_t_20260205-214507
```python
{
"precision": 0.7073170731707317,
"recall": 0.6170212765957447,
"f1": 0.6590909090909091,
"auc": 0.7701684397163121,
"pr_auc": 0.7517273411868727,
"fpr": 0.125,
"tpr": 0.6170212765957447,
"R": 0.4920212765957447,
"conf": 1.0,
"threshold": 0.49408891797065735
}
```
AZDX+b: checkpoints/clf_mixer_attnpl_t_20260205-213430
```python
{
"precision": 0.9090909090909091,
"recall": 0.6382978723404256,
"f1": 0.75,
"auc": 0.8993794326241135,
"pr_auc": 0.8516222215159499,
"fpr": 0.03125,
"tpr": 0.6382978723404256,
"R": 0.6070478723404256,
"conf": 1.0,
"threshold": 0.47319287061691284
}
```
SCEDC-b: checkpoints/clf_mixer_attnpl_t_20260205-215511
```python
{
"precision": 0.8484848484848485,
"recall": 0.5957446808510638,
"f1": 0.7,
"auc": 0.8089539007092198,
"pr_auc": 0.7909906637915577,
"fpr": 0.052083333333333336,
"tpr": 0.5957446808510638,
"R": 0.5436613475177304,
"conf": 1.0,
"threshold": 0.5665202140808105
}
```
SCEDC+b: checkpoints/clf_mixer_attnpl_t_20260205-215943
```python
{
"precision": 0.6,
"recall": 0.8297872340425532,
"f1": 0.6964285714285714,
"auc": 0.8386524822695036,
"pr_auc": 0.7720096405304839,
"fpr": 0.2708333333333333,
"tpr": 0.8297872340425532,
"R": 0.5589539007092199,
"conf": 1.0,
"threshold": 0.36804884672164917
}
```


## 3. REG
lstm    checkpoints/lstm_20260214-022353
```python
{'RMSE': 0.7786785802338432, 'MAE': 0.630647586343067, 'MSE': 0.6063403313149938, 'MAPE': 13.349332920574374, 'R2': -0.8119467090974293, 'PearsonR': 0.041928458831528355, 'PearsonR_diff': -0.07191252288834421, 'DA': 0.038461538461538464, 'SpearmanR': 0.016054864188288234, 'SpearmanR_diff': -0.0700384080804724, 'DTW': 60.26078748703003, 'DTW_normalized': 0.32929392069415314}
```

reg    checkpoints/reg_mixer_attnpl_t_20260204-180531
```python
{
"RMSE": 0.5612240989461469,
"MAE": 0.47766214641716964,
"MSE": 0.3149724892379145,
"MAPE": 9.80141362503057,
"R2": 0.058756885116413526,
"PearsonR": 0.47103774322856995,
"PearsonR_diff": -0.035590161795490544,
"DA": 0.02197802197802198,
"SpearmanR": 0.4502535274499076,
"SpearmanR_diff": -0.10597128141146665,
"DTW": 49.449580669403076,
"DTW_normalized": 0.27021628781094575
}
```
+log-pretrain    checkpoints/reg_mixer_attnpl_t_20260204-195612
```python
{
"RMSE": 0.6967193697000985,
"MAE": 0.5560744108398104,
"MSE": 0.4854178801153026,
"MAPE": 12.103934443682899,
"R2": -0.4505909344825294,
"PearsonR": 0.3561352046655124,
"PearsonR_diff": -0.040748607418345545,
"DA": 0.054945054945054944,
"SpearmanR": 0.2821346146611143,
"SpearmanR_diff": 0.025920564786629557,
"DTW": 70.19533920288086,
"DTW_normalized": 0.38358108854033257
}
```
-txpos   checkpoints/reg_mixer_attnpl_t_20260204-201607
```python
{
"RMSE": 0.8207982616353938,
"MAE": 0.6701326735032712,
"MSE": 0.6737097863036845,
"MAPE": 14.327793664108448,
"R2": -1.0132701091524514,
"PearsonR": 0.082520788323791,
"PearsonR_diff": 0.097945228273251,
"DA": 0.07142857142857142,
"SpearmanR": 0.2948227030910108,
"SpearmanR_diff": 0.1541373571502177,
"DTW": 67.44506025314331,
"DTW_normalized": 0.36855224182045526
}
```
-TMAP   checkpoints/reg_mixer_attnpl_t_20260204-194201
```python
{
"RMSE": 0.8604653763278692,
"MAE": 0.6800689723322301,
"MSE": 0.7404006638590617,
"MAPE": 14.565757840358703,
"R2": -1.2125647506508965,
"PearsonR": -0.1270456181669172,
"PearsonR_diff": 0.004238936123854664,
"DA": 0.04945054945054945,
"SpearmanR": -0.15318033772607112,
"SpearmanR_diff": 0.002720801658595025,
"DTW": 98.0214409828186,
"DTW_normalized": 0.5356362895235989
}
```
-pretrain checkpoints/reg_mixer_attnpl_t_20260204-192629
```python
{
"RMSE": 0.8227494716627346,
"MAE": 0.667753959614071,
"MSE": 0.6769166931213089,
"MAPE": 14.39205608720289,
"R2": -1.0228534190137841,
"PearsonR": 0.048141865263960304,
"PearsonR_diff": 0.0481822152770546,
"DA": 0.04945054945054945,
"SpearmanR": -0.010435611976882649,
"SpearmanR_diff": 0.023302286089083038,
"DTW": 59.63339376449585,
"DTW_normalized": 0.3258655396966986
}
```


## 4. TPP

### 4.1 SCEDC
tpp:  checkpoints/mixer_tpp_20260204-233222
```python
{
"nll_train_time": -22.78180694580078,
"nll_train_mag": 1.7235289812088013,
"nll_train_total": -21.058277130126953,
"nll_val_time": -13.54190731048584,
"nll_val_mag": 2.3293392658233643,
"nll_val_total": -11.212568283081055,
"nll_test_time": -14.715378761291504,
"nll_test_mag": 1.9568595886230469,
"nll_test_total": -12.758519172668457,
"num_events_train": 78087,
"num_events_val": 22928
}
```
tpp+b:  checkpoints/mixer_tpp_20260205-110204 checkpoints/mixer_tpp_20260205-100346
```python
{
"nll_train_time": -22.822824478149414,
"nll_train_mag": 1.364594578742981,
"nll_train_total": -21.448881149291992,
"nll_train_b_smooth": 9.348902702331543,
"nll_val_time": -13.574907302856445,
"nll_val_mag": 2.29687762260437,
"nll_val_total": -11.2686128616333,
"nll_val_b_smooth": 9.417391777038574,
"nll_test_time": -14.719155311584473,
"nll_test_mag": 1.7448956966400146,
"nll_test_total": -6.77383279800415,
"nll_test_b_smooth": 6200.4267578125,
"num_events_train": 78087,
"num_events_val": 22928
}
```
tpp+log: checkpoints/mixer_tpp_20260205-113617
```python
{
"nll_train_time": -22.990386962890625,
"nll_train_mag": 1.7235289812088013,
"nll_train_total": -21.26685905456543,
"nll_val_time": -13.556245803833008,
"nll_val_mag": 2.3293392658233643,
"nll_val_total": -11.226905822753906,
"nll_test_time": -14.827214241027832,
"nll_test_mag": 1.9568595886230469,
"nll_test_total": -12.870354652404785,
"num_events_train": 78087,
"num_events_val": 22928
}
```

tpp+log+b: checkpoints/mixer_tpp_20260205-115028
```python
{
"nll_train_time": -22.985198974609375,
"nll_train_mag": 1.3462562561035156,
"nll_train_total": -21.62884521484375,
"nll_train_b_smooth": 10.09611701965332,
"nll_val_time": -13.57252311706543,
"nll_val_mag": 2.285372257232666,
"nll_val_total": -11.277849197387695,
"nll_val_b_smooth": 9.301961898803711,
"nll_test_time": -14.819453239440918,
"nll_test_mag": 1.7460901737213135,
"nll_test_total": -12.277070045471191,
"nll_test_b_smooth": 796.2927856445312,
"num_events_train": 78087,
"num_events_val": 22928
}
```
tpp-txpos:  checkpoints/mixer_tpp_20260205-120039
```python
{
"nll_train_time": -19.720436096191406,
"nll_train_mag": 1.4095942974090576,
"nll_train_total": -18.295913696289062,
"nll_train_b_smooth": 14.926164627075195,
"nll_val_time": -11.315781593322754,
"nll_val_mag": 2.349069356918335,
"nll_val_total": -8.953110694885254,
"nll_val_b_smooth": 13.601122856140137,
"nll_test_time": -12.08328628540039,
"nll_test_mag": 1.7744652032852173,
"nll_test_total": -3.0220870971679688,
"nll_test_b_smooth": 7286.73388671875,
"num_events_train": 78087,
"num_events_val": 22928
}
```

### 4.2 ChuanDian
recast: checkpoints/rtpp_20260206-160628
tpp:  checkpoints/mixer_tpp_20260206-102040
```python
{
"nll_train_time": 0.31568610668182373,
"nll_train_mag": 0.3903427720069885,
"nll_train_total": 0.7060288190841675,
"nll_val_time": 0.814254105091095,
"nll_val_mag": 0.10913971811532974,
"nll_val_total": 0.9233937859535217,
"nll_test_time": 0.5817070603370667,
"nll_test_mag": 0.2756337821483612,
"nll_test_total": 0.8573408722877502,
"num_events_train": 4498,
"num_events_val": 567
}
```
tpp+b:  checkpoints/mixer_tpp_20260204-225304
```python
{
"nll_train_time": 0.3222985863685608,
"nll_train_mag": 0.35153552889823914,
"nll_train_total": 0.6654345393180847,
"nll_train_b_smooth": 0.22112269699573517,
"nll_train_b": -0.8620726466178894,
"nll_val_time": 0.8121285438537598,
"nll_val_mag": 0.09252987056970596,
"nll_val_total": 0.9009965658187866,
"nll_val_b_smooth": 0.011793801560997963,
"nll_val_b": -0.3673640489578247,
"nll_test_time": 0.5788850784301758,
"nll_test_mag": 0.257995069026947,
"nll_test_total": 0.8296654224395752,
"nll_test_b_smooth": 0.18049833178520203,
"nll_test_b": -0.7395214438438416,
"num_events_train": 4498,
"num_events_val": 567
}
```
tpp+b+pretrain:
tpp+log:  checkpoints/mixer_tpp_20260206-102404
```python
{
"nll_train_time": 0.38025331497192383,
"nll_train_mag": 0.3903427720069885,
"nll_train_total": 0.7705960273742676,
"nll_val_time": 0.8064956665039062,
"nll_val_mag": 0.10913971811532974,
"nll_val_total": 0.9156354069709778,
"nll_test_time": 0.5922229886054993,
"nll_test_mag": 0.2756337821483612,
"nll_test_total": 0.8678567409515381,
"num_events_train": 4498,
"num_events_val": 567
}
```
tpp+log+b: checkpoints/mixer_tpp_20260206-102755
```python
{
"nll_train_time": 0.381284236907959,
"nll_train_mag": 0.3532451093196869,
"nll_train_total": 0.7276188731193542,
"nll_train_b_smooth": 0.47875624895095825,
"nll_train_b": -0.7389227747917175,
"nll_val_time": 0.8054723739624023,
"nll_val_mag": 0.09412700682878494,
"nll_val_total": 0.8963361382484436,
"nll_val_b_smooth": 0.036167893558740616,
"nll_val_b": -0.3299429714679718,
"nll_test_time": 0.5828214287757874,
"nll_test_mag": 0.2611628472805023,
"nll_test_total": 0.8377600908279419,
"nll_test_b_smooth": 0.7203924655914307,
"nll_test_b": -0.6944623589515686,
"num_events_train": 4498,
"num_events_val": 567
}
```
tpp-txpos:  checkpoints/mixer_tpp_20260206-103125
```python
{
"nll_train_time": 0.38977503776550293,
"nll_train_mag": 0.3529645502567291,
"nll_train_total": 0.7354487180709839,
"nll_train_b_smooth": 0.5411959290504456,
"nll_train_b": -0.7832092046737671,
"nll_val_time": 0.8174117207527161,
"nll_val_mag": 0.09343661367893219,
"nll_val_total": 0.9077950119972229,
"nll_val_b_smooth": 0.020484743639826775,
"nll_val_b": -0.3073813021183014,
"nll_test_time": 0.6853238940238953,
"nll_test_mag": 0.2617284655570984,
"nll_test_total": 0.941096305847168,
"nll_test_b_smooth": 0.40408560633659363,
"nll_test_b": -0.6360194087028503,
"num_events_train": 4498,
"num_events_val": 567
}
```





### 4.3 T-XPOS Ablation

swc
2*Transformer+T-XPOS:  checkpoints/mixer_tpp_20260206-104534
```python
{
"nll_train_time": 0.33136501908302307,
"nll_train_mag": 0.35331791639328003,
"nll_train_total": 0.6776658296585083,
"nll_train_b_smooth": 0.6837563514709473,
"nll_train_b": -0.770089864730835,
"nll_val_time": 0.8136447072029114,
"nll_val_mag": 0.09093339741230011,
"nll_val_total": 0.9007568955421448,
"nll_val_b_smooth": 0.015644114464521408,
"nll_val_b": -0.3836888074874878,
"nll_test_time": 0.5791699290275574,
"nll_test_mag": 0.25999686121940613,
"nll_test_total": 0.8330843448638916,
"nll_test_b_smooth": 0.9895867705345154,
"nll_test_b": -0.7071958780288696,
"num_events_train": 4498,
"num_events_val": 567
}
```
2*Transformer+T-RoPE:  checkpoints/mixer_tpp_20260206-105017
```python
{
"nll_train_time": 0.7072980999946594,
"nll_train_mag": 0.4181928038597107,
"nll_train_total": 1.1515631675720215,
"nll_train_b_smooth": 0.9006957411766052,
"nll_train_b": 2.517158269882202,
"nll_val_time": 0.846195638179779,
"nll_val_mag": 0.0903928279876709,
"nll_val_total": 0.9327123165130615,
"nll_val_b_smooth": 0.03207719698548317,
"nll_val_b": -0.39082375168800354,
"nll_test_time": 0.8092707395553589,
"nll_test_mag": 0.27088141441345215,
"nll_test_total": 1.0788803100585938,
"nll_test_b_smooth": 0.9315567016601562,
"nll_test_b": -0.22034844756126404,
"num_events_train": 4498,
"num_events_val": 567
}
```
2*Transformer+XPOS:  checkpoints/mixer_tpp_20260206-105722
```python
{
"nll_train_time": 0.3651009500026703,
"nll_train_mag": 0.353091835975647,
"nll_train_total": 0.710474967956543,
"nll_train_b_smooth": 0.8548386096954346,
"nll_train_b": -0.8572676777839661,
"nll_val_time": 0.8170661330223083,
"nll_val_mag": 0.09409298747777939,
"nll_val_total": 0.9080309867858887,
"nll_val_b_smooth": 0.04528118669986725,
"nll_val_b": -0.31733882427215576,
"nll_test_time": 0.7830753922462463,
"nll_test_mag": 0.2623463571071625,
"nll_test_total": 1.039358377456665,
"nll_test_b_smooth": 0.7982481122016907,
"nll_test_b": -0.6861628293991089,
"num_events_train": 4498,
"num_events_val": 567
}
```
2*Transformer+RoPE:  checkpoints/mixer_tpp_20260206-105441
```python
{
"nll_train_time": 0.7065466642379761,
"nll_train_mag": 0.41747280955314636,
"nll_train_total": 1.1501120328903198,
"nll_train_b_smooth": 1.0857826471328735,
"nll_train_b": 2.500666618347168,
"nll_val_time": 0.846435546875,
"nll_val_mag": 0.09012433141469955,
"nll_val_total": 0.9326537847518921,
"nll_val_b_smooth": 0.035654764622449875,
"nll_val_b": -0.3941717743873596,
"nll_test_time": 0.8091757893562317,
"nll_test_mag": 0.27161338925361633,
"nll_test_total": 1.0799651145935059,
"nll_test_b_smooth": 0.993423581123352,
"nll_test_b": -0.1817438155412674,
"num_events_train": 4498,
"num_events_val": 567
}
```
None: checkpoints/mixer_tpp_20260206-170240
```python
{
"nll_train_time": 0.5658125877380371,
"nll_train_mag": 0.38859450817108154,
"nll_train_total": 0.96639484167099,
"nll_train_b_smooth": 1.3839315176010132,
"nll_train_b": 1.0603855848312378,
"nll_val_time": 0.8296405076980591,
"nll_val_mag": 0.09121473878622055,
"nll_val_total": 0.9171067476272583,
"nll_val_b_smooth": 0.05672850459814072,
"nll_val_b": -0.38052812218666077,
"nll_test_time": 0.7751372456550598,
"nll_test_mag": 0.2664538323879242,
"nll_test_total": 1.0387595891952515,
"nll_test_b_smooth": 0.9280709028244019,
"nll_test_b": -0.3759616017341614,
"num_events_train": 4498,
"num_events_val": 567
}
```


## 4.5  Ablations
 Mamaba-Transformer-Mamba+b: checkpoints/mixer_tpp_20260214-155713
 {'nll_train_time': 0.3222185969352722, 'nll_train_mag': 0.3515360355377197, 'nll_train_total': 0.6653570532798767, 'nll_train_b_smooth': 0.22093313932418823, 'nll_train_b': -0.8618493676185608, 'nll_val_time': 0.8121242523193359, 'nll_val_mag': 0.09252794831991196, 'nll_val_total': 0.900989830493927, 'nll_val_b_smooth': 0.011737565509974957, 'nll_val_b': -0.3674156069755554, 'nll_test_time': 0.5787126421928406, 'nll_test_mag': 0.257999062538147, 'nll_test_total': 0.8294956684112549, 'nll_test_b_smooth': 0.17975646257400513, 'nll_test_b': -0.7395869493484497, 'num_events_train': 4498, 'num_events_val': 567, 'num_events_test': 1043}
Mamaba-Transformer+b: checkpoints/mixer_tpp_20260214-160212
{'nll_train_time': 0.3467836380004883, 'nll_train_mag': 0.35419711470603943, 'nll_train_total': 0.6933996677398682, 'nll_train_b_smooth': 0.22821620106697083, 'nll_train_b': -0.7809317708015442, 'nll_val_time': 0.8140517473220825, 'nll_val_mag': 0.09297887980937958, 'nll_val_total': 0.9037079811096191, 'nll_val_b_smooth': 0.009493366815149784, 'nll_val_b': -0.33320996165275574, 'nll_test_time': 0.5956902503967285, 'nll_test_mag': 0.2581000328063965, 'nll_test_total': 0.8473722338676453, 'nll_test_b_smooth': 0.30752190947532654, 'nll_test_b': -0.6725601553916931, 'num_events_train': 4498, 'num_events_val': 567, 'num_events_test': 1043}
Mamba+b: 
 {'nll_train_time': 0.38333022594451904, 'nll_train_mag': 0.3550114035606384, 'nll_train_total': 0.7324224710464478, 'nll_train_b_smooth': 0.35067859292030334, 'nll_train_b': -0.6269875168800354, 'nll_val_time': 0.81813645362854, 'nll_val_mag': 0.09139693528413773, 'nll_val_total': 0.9056593179702759, 'nll_val_b_smooth': 0.022382037714123726, 'nll_val_b': -0.38964638113975525, 'nll_test_time': 0.6036948561668396, 'nll_test_mag': 0.2565937638282776, 'nll_test_total': 0.8543782830238342, 'nll_test_b_smooth': 0.47344064712524414, 'nll_test_b': -0.6383724808692932, 'num_events_train': 4498, 'num_events_val': 567, 'num_events_test': 1043}
 Mamaba-Transformer-2Mamba+b: checkpoints/mixer_tpp_20260214-161022
 {'nll_train_time': 0.3067726790904999, 'nll_train_mag': 0.3506609797477722, 'nll_train_total': 0.6487233638763428, 'nll_train_b_smooth': 0.3302916884422302, 'nll_train_b': -0.9040578603744507, 'nll_val_time': 0.8086276650428772, 'nll_val_mag': 0.09409730136394501, 'nll_val_total': 0.8995781540870667, 'nll_val_b_smooth': 0.015538553707301617, 'nll_val_b': -0.31623610854148865, 'nll_test_time': 0.5846673846244812, 'nll_test_mag': 0.2593567967414856, 'nll_test_total': 0.8378639221191406, 'nll_test_b_smooth': 0.30819666385650635, 'nll_test_b': -0.6468420028686523, 'num_events_train': 4498, 'num_events_val': 567, 'num_events_test': 1043}

### 4.4  Forecst Catalog Generation

SCEDC：checkpoints/mixer_tpp_20260206-112901
../checkpoints/mixer_tpp_20260205-111733

ChuanDian：checkpoints/mixer_tpp_20260207-095619
checkpoints/mixer_tpp_20260204-220105