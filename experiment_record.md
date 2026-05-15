# Experiment Record

## 1. CLF

**Config fields:** `dataset | Twindow | Tfore | dt | Mf | context_len | p/n`

### 1.1 ChuanDian 180 10 10 4 1 1

Class balance:

| Split | Positive | Negative | Pos/Neg | Positive ratio | Negative ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 780 | 683 | 1.14 | 53.33% | 46.67% |
| Train | 594 | 576 | 1.03 | 50.77% | 49.23% |
| Validation | 107 | 39 | 2.74 | 73.29% | 26.71% |
| Test | 79 | 68 | 1.16 | 53.74% | 46.26% |


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

### 1.2 ChuanDian 180 20 10 4.5 1 1

Class balance:

| Split | Positive | Negative | Pos/Neg | Positive ratio | Negative ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 623 | 845 | 0.74 | 42.43% | 57.57% |
| Train | 528 | 646 | 0.82 | 44.99% | 55.01% |
| Validation | 60 | 86 | 0.70 | 41.10% | 58.90% |
| Test | 35 | 113 | 0.31 | 23.65% | 76.35% |

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

### 1.3 ChuanDian 180 30 10 4.5 1 1.5 ~

Class balance:

| Split | Positive | Negative | Pos/Neg | Positive ratio | Negative ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 836 | 594 | 1.41 | 58.44% | 41.56% |
| Train | 706 | 438 | 1.61 | 61.68% | 38.32% |
| Validation | 83 | 60 | 1.38 | 58.04% | 41.96% |
| Test | 47 | 96 | 0.49 | 32.87% | 67.13% |

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

### 1.4 ChuanDian 180 60 10 5 1 1 ~

Class balance:

| Split | Positive | Negative | Pos/Neg | Positive ratio | Negative ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 808 | 651 | 1.24 | 55.38% | 44.62% |
| Train | 673 | 494 | 1.36 | 57.69% | 42.31% |
| Validation | 87 | 58 | 1.50 | 60.00% | 40.00% |
| Test | 48 | 99 | 0.48 | 32.65% | 67.35% |


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

### 1.5 ChuanDian 180 90 10 5.5 1 0.5 ~

Class balance:

| Split | Positive | Negative | Pos/Neg | Positive ratio | Negative ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 471 | 1055 | 0.45 | 30.87% | 69.13% |
| Train | 390 | 830 | 0.47 | 31.97% | 68.03% |
| Validation | 62 | 90 | 0.69 | 40.79% | 59.21% |
| Test | 19 | 135 | 0.14 | 12.34% | 87.66% |

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
{"precision": 0.717391304347826, "recall": 0.7021276595744681, "f1": 0.7096774193548387, "auc": 0.8353280141843972, "pr_auc": 0.8148300015050373, "fpr": 0.13541666666666666, "tpr": 0.7021276595744681, "R": 0.5667109929078015, "conf": 1.0, "threshold": 0.37369078397750854}
```

AZDX-b：checkpoints/clf_mixer_attnpl_t_20260205-213729 checkpoints/clf_mixer_attnpl_t_20260205-214507

```python
{"precision": 0.7073170731707317, "recall": 0.6170212765957447, "f1": 0.6590909090909091, "auc": 0.7701684397163121, "pr_auc": 0.7517273411868727, "fpr": 0.125, "tpr": 0.6170212765957447, "R": 0.4920212765957447, "conf": 1.0, "threshold": 0.49408891797065735}
```

AZDX+b: checkpoints/clf_mixer_attnpl_t_20260205-213430

```python
{"precision": 0.9090909090909091, "recall": 0.6382978723404256, "f1": 0.75, "auc": 0.8993794326241135, "pr_auc": 0.8516222215159499, "fpr": 0.03125, "tpr": 0.6382978723404256, "R": 0.6070478723404256, "conf": 1.0, "threshold": 0.47319287061691284}
```

SCEDC-b: checkpoints/clf_mixer_attnpl_t_20260205-215511

```python
{"precision": 0.8484848484848485, "recall": 0.5957446808510638, "f1": 0.7, "auc": 0.8089539007092198, "pr_auc": 0.7909906637915577, "fpr": 0.052083333333333336, "tpr": 0.5957446808510638, "R": 0.5436613475177304, "conf": 1.0, "threshold": 0.5665202140808105}
```

SCEDC+b: checkpoints/clf_mixer_attnpl_t_20260205-215943

```python
{"precision": 0.6, "recall": 0.8297872340425532, "f1": 0.6964285714285714, "auc": 0.8386524822695036, "pr_auc": 0.7720096405304839, "fpr": 0.2708333333333333, "tpr": 0.8297872340425532, "R": 0.5589539007092199, "conf": 1.0, "threshold": 0.36804884672164917}
```

## 3. REG

lstm checkpoints/lstm_20260510-212317

```python
{
  "RMSE": 0.7210919343517819,
  "MAE": 0.6223878964700335,
  "MSE": 0.5199735777871944,
  "MAPE": 13.277005674812628,
  "R2": -0.5538549134807047,
  "PearsonR": -0.009861617390003267,
  "PearsonR_diff": -0.055625542991226254,
  "DA": 0.054945054945054944,
  "SpearmanR": 0.10756370991216424,
  "SpearmanR_diff": -0.034333362378821564,
  "DTW": 75.70545864105225,
  "DTW_normalized": 0.41369103082542213
}
```

reg experiments/reg_mixer_attnpl_a_b/runs/reg_ab_reg_seed_1

```python
{
  "RMSE": 0.5813048198937832,
  "MAE": 0.49367417517255563,
  "MSE": 0.3379152936317437,
  "MAPE": 10.63200524839055,
  "R2": -0.009803689224253231,
  "PearsonR": 0.5769702822740322,
  "PearsonR_diff": 0.10149580215245169,
  "DA": 0.06593406593406594,
  "SpearmanR": 0.531315817185866,
  "SpearmanR_diff": 0.11509549782679587,
  "DTW": 66.2691179215908,
  "DTW_normalized": 0.3621263274403869
}
```

+log-pretrain checkpoints/reg_mixer_attnpl_t_20260204-195612

```python
{"RMSE": 0.6967193697000985, "MAE": 0.5560744108398104, "MSE": 0.4854178801153026, "MAPE": 12.103934443682899, "R2": -0.4505909344825294, "PearsonR": 0.3561352046655124, "PearsonR_diff": -0.040748607418345545, "DA": 0.054945054945054944, "SpearmanR": 0.2821346146611143, "SpearmanR_diff": 0.025920564786629557, "DTW": 70.19533920288086, "DTW_normalized": 0.38358108854033257}
```

-txpos checkpoints/reg_mixer_attnpl_t_20260204-201607

```python
{"RMSE": 0.8207982616353938, "MAE": 0.6701326735032712, "MSE": 0.6737097863036845, "MAPE": 14.327793664108448, "R2": -1.0132701091524514, "PearsonR": 0.082520788323791, "PearsonR_diff": 0.097945228273251, "DA": 0.07142857142857142, "SpearmanR": 0.2948227030910108, "SpearmanR_diff": 0.1541373571502177, "DTW": 67.44506025314331, "DTW_normalized": 0.36855224182045526}
```

-TMAP checkpoints/reg_mixer_attnpl_t_20260204-194201

```python
{"RMSE": 0.8604653763278692, "MAE": 0.6800689723322301, "MSE": 0.7404006638590617, "MAPE": 14.565757840358703, "R2": -1.2125647506508965, "PearsonR": -0.1270456181669172, "PearsonR_diff": 0.004238936123854664, "DA": 0.04945054945054945, "SpearmanR": -0.15318033772607112, "SpearmanR_diff": 0.002720801658595025, "DTW": 98.0214409828186, "DTW_normalized": 0.5356362895235989}
```

-pretrain checkpoints/reg_mixer_attnpl_t_20260204-192629

```python
{"RMSE": 0.8227494716627346, "MAE": 0.667753959614071, "MSE": 0.6769166931213089, "MAPE": 14.39205608720289, "R2": -1.0228534190137841, "PearsonR": 0.048141865263960304, "PearsonR_diff": 0.0481822152770546, "DA": 0.04945054945054945, "SpearmanR": -0.010435611976882649, "SpearmanR_diff": 0.023302286089083038, "DTW": 59.63339376449585, "DTW_normalized": 0.3258655396966986}
```

## 4. TPP

### 4.1 SCEDC
etas: checkpoints/etas_20260507-144632
{
  "nll_train_time": -22.22336196899414,
  "nll_train_total": -22.22336196899414,
  "nll_val_time": -13.204398155212402,
  "nll_val_total": -13.204398155212402,
  "nll_test_time": -14.634963035583496,
  "nll_test_total": -14.634963035583496,
  "num_events_train": 78087,
  "num_events_val": 22928,
  "num_events_test": 12806
}
recast: checkpoints/rtpp_20260506-120628
{
  "nll_train_time": -22.22336196899414,
  "nll_train_total": -22.22336196899414,
  "nll_val_time": -13.204398155212402,
  "nll_val_total": -13.204398155212402,
  "nll_test_time": -14.634963035583496,
  "nll_test_total": -14.634963035583496,
  "num_events_train": 78087,
  "num_events_val": 22928,
  "num_events_test": 12806
}
tpp: checkpoints/mixer_tpp_20260204-233222

```python
{"nll_train_time": -22.78180694580078, "nll_train_mag": 1.7235289812088013, "nll_train_total": -21.058277130126953, "nll_val_time": -13.54190731048584, "nll_val_mag": 2.3293392658233643, "nll_val_total": -11.212568283081055, "nll_test_time": -14.715378761291504, "nll_test_mag": 1.9568595886230469, "nll_test_total": -12.758519172668457, "num_events_train": 78087, "num_events_val": 22928}
```

tpp+b: checkpoints/mixer_tpp_20260205-110204 checkpoints/mixer_tpp_20260205-100346

```python
{
  "nll_train_time": -22.82278823852539,
  "nll_train_mag": 1.3645894527435303,
  "nll_train_total": -21.4488468170166,
  "nll_train_b_smooth": 9.350515365600586,
  "nll_val_time": -13.574847221374512,
  "nll_val_mag": 2.2968695163726807,
  "nll_val_total": -11.26855754852295,
  "nll_val_b_smooth": 9.420469284057617,
  "nll_test_time": -14.701654434204102,
  "nll_test_mag": 1.738781452178955,
  "nll_test_total": -1.759217381477356,
  "nll_test_b_smooth": 11203.6552734375,
  "num_events_train": 78087,
  "num_events_val": 22928,
  "num_events_test": 12806
}
```

tpp+log: checkpoints/mixer_tpp_20260205-113617

```python
{"nll_train_time": -22.990386962890625, "nll_train_mag": 1.7235289812088013, "nll_train_total": -21.26685905456543, "nll_val_time": -13.556245803833008, "nll_val_mag": 2.3293392658233643, "nll_val_total": -11.226905822753906, "nll_test_time": -14.827214241027832, "nll_test_mag": 1.9568595886230469, "nll_test_total": -12.870354652404785, "num_events_train": 78087, "num_events_val": 22928}
```

tpp+log+b: checkpoints/mixer_tpp_20260205-115028

```python
{"nll_train_time": -22.985198974609375, "nll_train_mag": 1.3462562561035156, "nll_train_total": -21.62884521484375, "nll_train_b_smooth": 10.09611701965332, "nll_val_time": -13.57252311706543, "nll_val_mag": 2.285372257232666, "nll_val_total": -11.277849197387695, "nll_val_b_smooth": 9.301961898803711, "nll_test_time": -14.819453239440918, "nll_test_mag": 1.7460901737213135, "nll_test_total": -12.277070045471191, "nll_test_b_smooth": 796.2927856445312, "num_events_train": 78087, "num_events_val": 22928}
```

tpp-txpos: checkpoints/mixer_tpp_20260205-120039

```python
{"nll_train_time": -19.720436096191406, "nll_train_mag": 1.4095942974090576, "nll_train_total": -18.295913696289062, "nll_train_b_smooth": 14.926164627075195, "nll_val_time": -11.315781593322754, "nll_val_mag": 2.349069356918335, "nll_val_total": -8.953110694885254, "nll_val_b_smooth": 13.601122856140137, "nll_test_time": -12.08328628540039, "nll_test_mag": 1.7744652032852173, "nll_test_total": -3.0220870971679688, "nll_test_b_smooth": 7286.73388671875, "num_events_train": 78087, "num_events_val": 22928}
```

### 4.2 ChuanDian

tpp: experiments/mixer_tpp_ablation_chuandian/runs/mixer_tpp_ablation_tpp_seed_0/best_model_1.pth

```python
{"nll_train_time": 0.3475324809551239, "nll_train_mag": 0.39034321904182434, "nll_train_total": 0.7378756999969482, "nll_val_time": 0.8152410984039307, "nll_val_mag": 0.10914000868797302, "nll_val_total": 0.9243811368942261, "nll_test_time": 0.5813219547271729, "nll_test_mag": 0.2756342589855194, "nll_test_total": 0.8569562435150146, "num_events_train": 4498, "num_events_val": 567, "num_events_test": 1043}
```

tpp+b: experiments/mixer_tpp_ablation_chuandian/runs/mixer_tpp_ablation_tpp_b_seed_0/best_model_1.pth

```python
{"nll_train_time": 0.3219917416572571, "nll_train_mag": 0.3520156145095825, "nll_train_total": 0.6656216979026794, "nll_train_b_smooth": 0.18635359406471252, "nll_train_b": -0.8571958541870117, "nll_val_time": 0.8137814998626709, "nll_val_mag": 0.09222788363695145, "nll_val_total": 0.9022765159606934, "nll_val_b_smooth": 0.023966442793607712, "nll_val_b": -0.3756851255893707, "nll_test_time": 0.5744537711143494, "nll_test_mag": 0.25969651341438293, "nll_test_total": 0.8274227976799011, "nll_test_b_smooth": 0.21125580370426178, "nll_test_b": -0.6938709020614624, "num_events_train": 4498, "num_events_val": 567, "num_events_test": 1043}
```

tpp+log: experiments/mixer_tpp_ablation_chuandian/runs/mixer_tpp_ablation_tpp_log_seed_0/best_model_1.pth

```python
{"nll_train_time": 0.4325093924999237, "nll_train_mag": 0.39034321904182434, "nll_train_total": 0.8228525519371033, "nll_val_time": 0.8052669763565063, "nll_val_mag": 0.10914000868797302, "nll_val_total": 0.9144070148468018, "nll_test_time": 0.6256822943687439, "nll_test_mag": 0.2756342589855194, "nll_test_total": 0.9013165235519409, "num_events_train": 4498, "num_events_val": 567, "num_events_test": 1043}
```

tpp+log+b: experiments/mixer_tpp_ablation_chuandian/runs/mixer_tpp_ablation_tpp_log_b_seed_0/best_model_1.pth

```python
{"nll_train_time": 0.3815753161907196, "nll_train_mag": 0.3531482219696045, "nll_train_total": 0.7281173467636108, "nll_train_b_smooth": 0.5842517018318176, "nll_train_b": -0.7190460562705994, "nll_val_time": 0.8049465417861938, "nll_val_mag": 0.09279705584049225, "nll_val_total": 0.8940302133560181, "nll_val_b_smooth": 0.03686581179499626, "nll_val_b": -0.37502533197402954, "nll_test_time": 0.5860284566879272, "nll_test_mag": 0.26216739416122437, "nll_test_total": 0.8415156602859497, "nll_test_b_smooth": 0.6671037077903748, "nll_test_b": -0.7347274422645569, "num_events_train": 4498, "num_events_val": 567, "num_events_test": 1043}
```

tpp-txpos: experiments/mixer_tpp_ablation_chuandian/runs/mixer_tpp_ablation_tpp_minus_txpos_seed_0/best_model_1.pth

```python
{"nll_train_time": 0.42139536142349243, "nll_train_mag": 0.3516053557395935, "nll_train_total": 0.7654439210891724, "nll_train_b_smooth": 0.4160616099834442, "nll_train_b": -0.7972947359085083, "nll_val_time": 0.8198422789573669, "nll_val_mag": 0.09153085201978683, "nll_val_total": 0.9075899720191956, "nll_val_b_smooth": 0.017568623647093773, "nll_val_b": -0.3800642192363739, "nll_test_time": 0.6802210807800293, "nll_test_mag": 0.2608191967010498, "nll_test_total": 0.9349408745765686, "nll_test_b_smooth": 0.33272218704223633, "nll_test_b": -0.6432147026062012, "num_events_train": 4498, "num_events_val": 567, "num_events_test": 1043}
```


### 4.3 T-XPOS Ablation

swc

2\*Transformer+T-XPOS: checkpoints/mixer_tpp_20260206-104534

```python
{"nll_train_time": 0.33136501908302307, "nll_train_mag": 0.35331791639328003, "nll_train_total": 0.6776658296585083, "nll_train_b_smooth": 0.6837563514709473, "nll_train_b": -0.770089864730835, "nll_val_time": 0.8136447072029114, "nll_val_mag": 0.09093339741230011, "nll_val_total": 0.9007568955421448, "nll_val_b_smooth": 0.015644114464521408, "nll_val_b": -0.3836888074874878, "nll_test_time": 0.5791699290275574, "nll_test_mag": 0.25999686121940613, "nll_test_total": 0.8330843448638916, "nll_test_b_smooth": 0.9895867705345154, "nll_test_b": -0.7071958780288696, "num_events_train": 4498, "num_events_val": 567}
```

2\*Transformer+T-RoPE: checkpoints/mixer_tpp_20260206-105017

```python
{"nll_train_time": 0.7072980999946594, "nll_train_mag": 0.4181928038597107, "nll_train_total": 1.1515631675720215, "nll_train_b_smooth": 0.9006957411766052, "nll_train_b": 2.517158269882202, "nll_val_time": 0.846195638179779, "nll_val_mag": 0.0903928279876709, "nll_val_total": 0.9327123165130615, "nll_val_b_smooth": 0.03207719698548317, "nll_val_b": -0.39082375168800354, "nll_test_time": 0.8092707395553589, "nll_test_mag": 0.27088141441345215, "nll_test_total": 1.0788803100585938, "nll_test_b_smooth": 0.9315567016601562, "nll_test_b": -0.22034844756126404, "num_events_train": 4498, "num_events_val": 567}
```

2\*Transformer+XPOS: checkpoints/mixer_tpp_20260206-105722

```python
{"nll_train_time": 0.3651009500026703, "nll_train_mag": 0.353091835975647, "nll_train_total": 0.710474967956543, "nll_train_b_smooth": 0.8548386096954346, "nll_train_b": -0.8572676777839661, "nll_val_time": 0.8170661330223083, "nll_val_mag": 0.09409298747777939, "nll_val_total": 0.9080309867858887, "nll_val_b_smooth": 0.04528118669986725, "nll_val_b": -0.31733882427215576, "nll_test_time": 0.7830753922462463, "nll_test_mag": 0.2623463571071625, "nll_test_total": 1.039358377456665, "nll_test_b_smooth": 0.7982481122016907, "nll_test_b": -0.6861628293991089, "num_events_train": 4498, "num_events_val": 567}
```

2\*Transformer+RoPE: checkpoints/mixer_tpp_20260206-105441

```python
{"nll_train_time": 0.7065466642379761, "nll_train_mag": 0.41747280955314636, "nll_train_total": 1.1501120328903198, "nll_train_b_smooth": 1.0857826471328735, "nll_train_b": 2.500666618347168, "nll_val_time": 0.846435546875, "nll_val_mag": 0.09012433141469955, "nll_val_total": 0.9326537847518921, "nll_val_b_smooth": 0.035654764622449875, "nll_val_b": -0.3941717743873596, "nll_test_time": 0.8091757893562317, "nll_test_mag": 0.27161338925361633, "nll_test_total": 1.0799651145935059, "nll_test_b_smooth": 0.993423581123352, "nll_test_b": -0.1817438155412674, "num_events_train": 4498, "num_events_val": 567}
```

None: checkpoints/mixer_tpp_20260206-170240

```python
{"nll_train_time": 0.5658125877380371, "nll_train_mag": 0.38859450817108154, "nll_train_total": 0.96639484167099, "nll_train_b_smooth": 1.3839315176010132, "nll_train_b": 1.0603855848312378, "nll_val_time": 0.8296405076980591, "nll_val_mag": 0.09121473878622055, "nll_val_total": 0.9171067476272583, "nll_val_b_smooth": 0.05672850459814072, "nll_val_b": -0.38052812218666077, "nll_test_time": 0.7751372456550598, "nll_test_mag": 0.2664538323879242, "nll_test_total": 1.0387595891952515, "nll_test_b_smooth": 0.9280709028244019, "nll_test_b": -0.3759616017341614, "num_events_train": 4498, "num_events_val": 567}
```

### 4.4 Ablations

Mamaba-Transformer-Mamba+b: checkpoints/mixer_tpp_20260214-155713

```python
{'nll_train_time': 0.3222185969352722, 'nll_train_mag': 0.3515360355377197, 'nll_train_total': 0.6653570532798767, 'nll_train_b_smooth': 0.22093313932418823, 'nll_train_b': -0.8618493676185608, 'nll_val_time': 0.8121242523193359, 'nll_val_mag': 0.09252794831991196, 'nll_val_total': 0.900989830493927, 'nll_val_b_smooth': 0.011737565509974957, 'nll_val_b': -0.3674156069755554, 'nll_test_time': 0.5787126421928406, 'nll_test_mag': 0.257999062538147, 'nll_test_total': 0.8294956684112549, 'nll_test_b_smooth': 0.17975646257400513, 'nll_test_b': -0.7395869493484497, 'num_events_train': 4498, 'num_events_val': 567, 'num_events_test': 1043}
```

Mamaba-Transformer+b: checkpoints/mixer_tpp_20260214-160212

```python
{'nll_train_time': 0.3467836380004883, 'nll_train_mag': 0.35419711470603943, 'nll_train_total': 0.6933996677398682, 'nll_train_b_smooth': 0.22821620106697083, 'nll_train_b': -0.7809317708015442, 'nll_val_time': 0.8140517473220825, 'nll_val_mag': 0.09297887980937958, 'nll_val_total': 0.9037079811096191, 'nll_val_b_smooth': 0.009493366815149784, 'nll_val_b': -0.33320996165275574, 'nll_test_time': 0.5956902503967285, 'nll_test_mag': 0.2581000328063965, 'nll_test_total': 0.8473722338676453, 'nll_test_b_smooth': 0.30752190947532654, 'nll_test_b': -0.6725601553916931, 'num_events_train': 4498, 'num_events_val': 567, 'num_events_test': 1043}
```

Mamba+b:

```python
{'nll_train_time': 0.38333022594451904, 'nll_train_mag': 0.3550114035606384, 'nll_train_total': 0.7324224710464478, 'nll_train_b_smooth': 0.35067859292030334, 'nll_train_b': -0.6269875168800354, 'nll_val_time': 0.81813645362854, 'nll_val_mag': 0.09139693528413773, 'nll_val_total': 0.9056593179702759, 'nll_val_b_smooth': 0.022382037714123726, 'nll_val_b': -0.38964638113975525, 'nll_test_time': 0.6036948561668396, 'nll_test_mag': 0.2565937638282776, 'nll_test_total': 0.8543782830238342, 'nll_test_b_smooth': 0.47344064712524414, 'nll_test_b': -0.6383724808692932, 'num_events_train': 4498, 'num_events_val': 567, 'num_events_test': 1043}
```

Mamaba-Transformer-2Mamba+b: checkpoints/mixer_tpp_20260214-161022

```python
{'nll_train_time': 0.3067726790904999, 'nll_train_mag': 0.3506609797477722, 'nll_train_total': 0.6487233638763428, 'nll_train_b_smooth': 0.3302916884422302, 'nll_train_b': -0.9040578603744507, 'nll_val_time': 0.8086276650428772, 'nll_val_mag': 0.09409730136394501, 'nll_val_total': 0.8995781540870667, 'nll_val_b_smooth': 0.015538553707301617, 'nll_val_b': -0.31623610854148865, 'nll_test_time': 0.5846673846244812, 'nll_test_mag': 0.2593567967414856, 'nll_test_total': 0.8378639221191406, 'nll_test_b_smooth': 0.30819666385650635, 'nll_test_b': -0.6468420028686523, 'num_events_train': 4498, 'num_events_val': 567, 'num_events_test': 1043}
```

### 4.5 Forecast Catalog Generation

SCEDC：checkpoints/mixer_tpp_20260206-112901
../checkpoints/mixer_tpp_20260205-111733

ChuanDian：checkpoints/mixer_tpp_20260207-095619
checkpoints/mixer_tpp_20260204-220105


### 4.6 

1. PNR_1z:

ETAS(mu=0): checkpoints/etas_20260403-123804

```python
{"nll_train_time": -5.975103855133057, "nll_train_total": -5.975103855133057, "nll_val_time": -6.446558952331543, "nll_val_total": -6.446558952331543, "nll_test_time": -6.365136623382568, "nll_test_total": -6.365136623382568, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.9333333333333333, "mae": 172.5735, "rmse": 331.34805195267603, "lp_nb": -5.745138939724267, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260403-123804/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


ETAS: checkpoints/etas_20260403-204045

```python
{"nll_train_time": -5.987344741821289, "nll_train_total": -5.987344741821289, "nll_val_time": -6.466350078582764, "nll_val_total": -6.466350078582764, "nll_test_time": -6.384871482849121, "nll_test_total": -6.384871482849121, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.36666666666666664, "mae": 181.37890000000002, "rmse": 287.7162165410563, "lp_nb": -6.438656939295559, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260403-204045/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


ETAS+P: checkpoints/etas_20260330-231015

```python
{"nll_train_time": -6.000354290008545, "nll_train_total": -6.000354290008545, "nll_val_time": -6.524806022644043, "nll_val_total": -6.524806022644043, "nll_test_time": -6.367061614990234, "nll_test_total": -6.367061614990234, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.9, "mae": 89.75679999999998, "rmse": 176.33591172985726, "lp_nb": -5.63508261387014, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-231015/sliding_window_cache.npz", "sliding_loaded_from_cache": true}
```

branching_ratio(10): 0.8860726303288199

ETAS+conv_mlp: checkpoints/etas_20260330-231441

```python
{"nll_train_time": -5.991494178771973, "nll_train_total": -5.991494178771973, "nll_val_time": -6.4781012535095215, "nll_val_total": -6.4781012535095215, "nll_test_time": -6.373833656311035, "nll_test_total": -6.373833656311035, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.8666666666666667, "mae": 123.45896666666665, "rmse": 248.66930687059067, "lp_nb": -5.49318542280698, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-231441/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


ETAS+kernel(Gamma): checkpoints/etas_kernel_pnr_1z_20260407

```python
{"nll_train_time": -5.975950241088867, "nll_train_total": -5.975950241088867, "nll_val_time": -6.4459052085876465, "nll_val_total": -6.4459052085876465, "nll_test_time": -6.364109516143799, "nll_test_total": -6.364109516143799, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.9333333333333333, "mae": 161.20003333333335, "rmse": 315.97086204917696, "lp_nb": -5.640436706699134, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_kernel_pnr_1z_20260407/sliding_window_cache.npz", "sliding_loaded_from_cache": true}
```

branching_ratio(10): 0.9574215764754445

ETAS+mamba: checkpoints/etas_20260330-231612

```python
{"nll_train_time": -6.013334274291992, "nll_train_total": -6.013334274291992, "nll_val_time": -6.540414810180664, "nll_val_total": -6.540414810180664, "nll_test_time": -6.382467269897461, "nll_test_total": -6.382467269897461, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.8666666666666667, "mae": 88.67206666666668, "rmse": 174.09297063695593, "lp_nb": -5.434532283951722, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-231612/sliding_window_cache.npz", "sliding_loaded_from_cache": true}
```


branching_ratio(10): 0.8706774613662029

ETAS+mamba+gp: checkpoints/etas_20260330-232554 \*

```python
{"nll_train_time": -5.998855113983154, "nll_train_total": -5.956877708435059, "nll_train_bg_kl": 0.041977860033512115, "nll_val_time": -6.542019844055176, "nll_val_total": -6.433897018432617, "nll_val_bg_kl": 0.10812271386384964, "nll_test_time": -6.383713722229004, "nll_test_total": -6.230784893035889, "nll_test_bg_kl": 0.15292899310588837, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.8666666666666667, "mae": 80.29406666666667, "rmse": 160.7142135153785, "lp_nb": -5.9691333932437765, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-232554/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```

branching_ratio(10): 0.854715837013654

RECAST+P: checkpoints/rtpp_20260402-170440

```python
{"nll_train_time": -5.778769493103027, "nll_train_total": -5.869544982910156, "nll_train_bg": -0.09077509492635727, "nll_val_time": -6.11020040512085, "nll_val_total": -6.373103618621826, "nll_val_bg": -0.2629033029079437, "nll_test_time": -6.049722194671631, "nll_test_total": -6.150845527648926, "nll_test_bg": -0.10112347453832626, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.7666666666666667, "mae": 88.46536666666667, "rmse": 226.89517505337125, "lp_nb": -7.278010667469726, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260402-170440/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


RECAST+conv_mlp: checkpoints/rtpp_20260402-212205

```python
{"nll_train_time": -5.895451068878174, "nll_train_total": -5.970457553863525, "nll_train_bg": -0.07500695437192917, "nll_val_time": -6.225480079650879, "nll_val_total": -6.422011852264404, "nll_val_bg": -0.1965317279100418, "nll_test_time": -6.169680118560791, "nll_test_total": -6.228274345397949, "nll_test_bg": -0.058594394475221634, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.8333333333333334, "mae": 74.29663333333335, "rmse": 212.74977864939524, "lp_nb": -6.214915918530949, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260402-212205/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


RECAST+kernel(Gamma): checkpoints/rtpp_kernel_pnr_1z_20260407

```python
{"nll_train_time": -5.891434669494629, "nll_train_total": -5.9714179039001465, "nll_train_bg": -0.07998327910900116, "nll_val_time": -6.203549385070801, "nll_val_total": -6.2786865234375, "nll_val_bg": -0.07513731718063354, "nll_test_time": -6.156153678894043, "nll_test_total": -6.214244365692139, "nll_test_bg": -0.058091215789318085, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.7333333333333333, "mae": 74.41266666666667, "rmse": 201.72816460656486, "lp_nb": -6.516608702322753, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/rtpp_kernel_pnr_1z_20260407/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


RECAST+mamba: checkpoints/rtpp_20260402-160131

```python
{"nll_train_time": -5.751069068908691, "nll_train_total": -5.892369270324707, "nll_train_bg": -0.14130006730556488, "nll_val_time": -6.074784278869629, "nll_val_total": -6.468404293060303, "nll_val_bg": -0.3936195373535156, "nll_test_time": -6.017630577087402, "nll_test_total": -6.22796630859375, "nll_test_bg": -0.21033605933189392, "num_events_train": 3049, "num_events_val": 1204, "num_events_test": 850}
```

```python
{"status": "ok", "coverage": 0.8666666666666667, "mae": 70.8163, "rmse": 191.78162484685546, "lp_nb": -6.571120268242326, "num_windows": 30, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260402-160131/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


2. CB_HAB1a(-0.4):

ETAS(mu=0): checkpoints/etas_20260403-202516

```python
{"nll_train_time": -5.319025993347168, "nll_train_total": -5.319025993347168, "nll_val_time": -5.221223831176758, "nll_val_total": -5.221223831176758, "nll_test_time": -5.096417427062988, "nll_test_total": -5.096417427062988, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```

```python
{"status": "ok", "coverage": 0.8571428571428571, "mae": 595.2535238095238, "rmse": 1104.7653487270497, "lp_nb": -6.20322783992592, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260403-202516/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


ETAS: checkpoints/etas_20260403-203341

```python
{"nll_train_time": -5.322861194610596, "nll_train_total": -5.322861194610596, "nll_val_time": -5.220753192901611, "nll_val_total": -5.220753192901611, "nll_test_time": -5.103967189788818, "nll_test_total": -5.103967189788818, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```
sliding_window_eval_metrics: checkpoints/etas_20260403-203341/sliding_window_eval_metrics.json

```python
{"status": "ok", "coverage": 0.9523809523809523, "mae": 579.2222857142858, "rmse": 1051.621645918703, "lp_nb": -6.0182237919010095, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260403-203341/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


ETAS+P: checkpoints/etas_20260330-155945

```python
{"nll_train_time": -5.095571994781494, "nll_train_total": -5.095571994781494, "nll_val_time": -4.848620891571045, "nll_val_total": -4.848620891571045, "nll_test_time": -3.7083961963653564, "nll_test_total": -3.7083961963653564, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```

```python
{"status": "ok", "coverage": 0.23809523809523808, "mae": 279.94423809523806, "rmse": 481.95024992069364, "lp_nb": -31.354834041082796, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-155945/sliding_window_cache.npz", "sliding_loaded_from_cache": true}
```


ETAS+conv_mlp: checkpoints/etas_20260330-161827

```python
{"nll_train_time": -5.319257736206055, "nll_train_total": -5.319257736206055, "nll_val_time": -5.2232255935668945, "nll_val_total": -5.2232255935668945, "nll_test_time": -5.0861029624938965, "nll_test_total": -5.0861029624938965, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```

```python
{"status": "ok", "coverage": 0.8571428571428571, "mae": 259.4428571428571, "rmse": 451.03885635919386, "lp_nb": -6.169341595338941, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-161827/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


ETAS+kernel(Gamma): checkpoints/etas_kernel_cb_hab1a_20260407

```python
{"nll_train_time": -5.321628093719482, "nll_train_total": -5.321628093719482, "nll_val_time": -5.217365741729736, "nll_val_total": -5.217365741729736, "nll_test_time": -5.085361480712891, "nll_test_total": -5.085361480712891, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```

```python
{"status": "ok", "coverage": 0.9047619047619048, "mae": 299.5488095238095, "rmse": 495.33174528299236, "lp_nb": -5.437695049821424, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_kernel_cb_hab1a_20260407/sliding_window_cache.npz", "sliding_loaded_from_cache": true}
```


ETAS+mamba: checkpoints/etas_20260330-161340

```python
{"nll_train_time": -5.321388244628906, "nll_train_total": -5.321388244628906, "nll_val_time": -5.216009140014648, "nll_val_total": -5.216009140014648, "nll_test_time": -5.0885725021362305, "nll_test_total": -5.0885725021362305, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```

```python
{"status": "ok", "coverage": 0.9047619047619048, "mae": 302.50766666666675, "rmse": 582.5223545819222, "lp_nb": -5.507037104414327, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-161340/sliding_window_cache.npz", "sliding_loaded_from_cache": true}
```


ETAS+mamba+gp: checkpoints/etas_20260330-144040 \*

```python
{"nll_train_time": -5.328885555267334, "nll_train_total": -5.302903652191162, "nll_train_bg_kl": 0.02598220854997635, "nll_val_time": -5.206945896148682, "nll_val_total": -5.011465072631836, "nll_val_bg_kl": 0.1954808086156845, "nll_test_time": -5.100043773651123, "nll_test_total": -4.795830249786377, "nll_test_bg_kl": 0.3042137920856476, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```

```python
{"status": "ok", "coverage": 0.8571428571428571, "mae": 107.58966666666667, "rmse": 165.25299734231464, "lp_nb": -5.165810906722411, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-144040/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


RECAST+P: checkpoints/rtpp_20260403-100551

```python
{"nll_train_time": -5.162586212158203, "nll_train_total": -5.250383377075195, "nll_train_bg": -0.08779700845479965, "nll_val_time": -5.003657341003418, "nll_val_total": -5.044312000274658, "nll_val_bg": -0.04065511003136635, "nll_test_time": -4.913900375366211, "nll_test_total": -4.694487571716309, "nll_test_bg": 0.21941331028938293, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```

```python
{"status": "ok", "coverage": 0.47619047619047616, "mae": 111.85119047619052, "rmse": 163.05448065556848, "lp_nb": -8.84562597131683, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260403-100551/sliding_window_cache.npz", "sliding_loaded_from_cache": true}
```


RECAST+conv_mlp: checkpoints/rtpp_20260403-112025

```python
{"nll_train_time": -5.177980422973633, "nll_train_total": -5.257157802581787, "nll_train_bg": -0.07917758077383041, "nll_val_time": -5.024879455566406, "nll_val_total": -5.049865245819092, "nll_val_bg": -0.024985933676362038, "nll_test_time": -4.9088358879089355, "nll_test_total": -4.726556777954102, "nll_test_bg": 0.18227897584438324, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```

```python
{"status": "ok", "coverage": 0.5238095238095238, "mae": 87.74485714285714, "rmse": 122.23259363898929, "lp_nb": -7.0608470752606465, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260403-112025/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


RECAST+kernel(Gamma): checkpoints/rtpp_kernel_cb_hab1a_20260407

```python
{"nll_train_time": -5.091848373413086, "nll_train_total": -5.24493932723999, "nll_train_bg": -0.1530907154083252, "nll_val_time": -4.935190200805664, "nll_val_total": -5.048030376434326, "nll_val_bg": -0.11284016072750092, "nll_test_time": -4.725852012634277, "nll_test_total": -4.654197692871094, "nll_test_bg": 0.07165437191724777, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```
```python
{"status": "ok", "coverage": 0.5714285714285714, "mae": 89.89099999999999, "rmse": 126.10299549989928, "lp_nb": -7.307970963899969, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/rtpp_kernel_cb_hab1a_20260407/sliding_window_cache.npz", "sliding_loaded_from_cache": true}
```


RECAST+mamba: checkpoints/rtpp_20260403-112626

```python
{"nll_train_time": -5.039740562438965, "nll_train_total": -5.240628719329834, "nll_train_bg": -0.2008877992630005, "nll_val_time": -4.890742301940918, "nll_val_total": -5.1092305183410645, "nll_val_bg": -0.21848855912685394, "nll_test_time": -4.797508239746094, "nll_test_total": -4.114445209503174, "nll_test_bg": 0.6830629110336304, "num_events_train": 5495, "num_events_val": 734, "num_events_test": 472}
```

```python
{"status": "ok", "coverage": 0.7142857142857143, "mae": 92.69466666666668, "rmse": 171.46136319632728, "lp_nb": -9.104251985931986, "num_windows": 21, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260403-112626/sliding_window_cache.npz", "sliding_loaded_from_cache": true}
```


3. CB_HAB4:

ETAS: checkpoints/etas_20260404-111418

```python
{"nll_train_time": -4.878288269042969, "nll_train_total": -4.878288269042969, "nll_val_time": -5.741217136383057, "nll_val_total": -5.741217136383057, "nll_test_time": -4.515438079833984, "nll_test_total": -4.515438079833984, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

ETAS+P: checkpoints/etas_20260330-183028

```python
{"nll_train_time": -4.635933876037598, "nll_train_total": -4.635933876037598, "nll_val_time": -5.55360746383667, "nll_val_total": -5.55360746383667, "nll_test_time": -3.8005523681640625, "nll_test_total": -3.8005523681640625, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

```python
{"status": "ok", "coverage": 0.0, "mae": 210.98711111111112, "rmse": 227.93734202188108, "lp_nb": -25.19578300867526, "num_windows": 9, "settings": {"duration": 2.0, "step": 2.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-183028/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```


ETAS+conv_mlp: checkpoints/etas_20260330-183526 checkpoints/etas_20260330-184009 (点太多，采样失败)

```python
{"nll_train_time": -4.907235622406006, "nll_train_total": -4.907235622406006, "nll_val_time": -5.7752909660339355, "nll_val_total": -5.7752909660339355, "nll_test_time": -4.563719272613525, "nll_test_total": -4.563719272613525, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

ETAS+kernel(Gamma): checkpoints/etas_20260409-215531 checkpoints/etas_kernel_cb_hab4_20260407
```python
{"nll_train_time": -5.171771049499512, "nll_train_total": -5.171771049499512, "nll_val_time": -6.008909225463867, "nll_val_total": -6.008909225463867, "nll_test_time": -4.784082889556885, "nll_test_total": -4.784082889556885, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

```python
{"status": "ok", "coverage": 0.7894736842105263, "mae": 73.31705263157895, "rmse": 95.71814546887126, "lp_nb": -5.822384701713899, "num_windows": 19, "settings": {"duration": 1.0, "step": 1.0, "quantiles": [2.5, 97.5], "samples_per_batch": 1000, "view_mode": "zoomed", "cache_filename": "sliding_window_cache.npz", "load_sliding_cache": true, "force_recompute_sliding": false}, "sliding_cache_path": "/root/autodl-tmp/em_eqf/checkpoints/etas_20260409-215531/sliding_window_cache.npz", "sliding_loaded_from_cache": false}
```

ETAS+mamba: checkpoints/etas_20260330-182010 \*

```python
{"nll_train_time": -5.182605743408203, "nll_train_total": -5.182605743408203, "nll_val_time": -6.006970405578613, "nll_val_total": -6.006970405578613, "nll_test_time": -4.775404930114746, "nll_test_total": -4.775404930114746, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

```python
{"status":"ok","coverage":0.8421052631578947,"mae":75.5101052631579,"rmse":97.03540041437942,"lp_nb":-5.719695003983212,"num_windows":19,"settings":{"duration":1,"step":1,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-182010/sliding_window_cache.npz","sliding_loaded_from_cache":false}
```

ETAS+mamba+gp: checkpoints/etas_20260330-185654

```python
{"nll_train_time": -5.183523654937744, "nll_train_total": -5.13915491104126, "nll_train_bg_kl": 0.04436931386590004, "nll_val_time": -6.001755237579346, "nll_val_total": -5.871023178100586, "nll_val_bg_kl": 0.1307324320077896, "nll_test_time": -4.739864349365234, "nll_test_total": -4.555238246917725, "nll_test_bg_kl": 0.18462617695331573, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

RECAST+P: checkpoints/rtpp_20260403-210053

```python
{"nll_train_time": -4.755245685577393, "nll_train_total": -5.148390769958496, "nll_train_bg": -0.3931451737880707, "nll_val_time": -5.656795501708984, "nll_val_total": -5.840083122253418, "nll_val_bg": -0.18328756093978882, "nll_test_time": -4.264465808868408, "nll_test_total": -4.33951473236084, "nll_test_bg": -0.07504881918430328, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

```python
{"status":"ok","coverage":0.631578947368421,"mae":83.25742105263159,"rmse":108.48330535811054,"lp_nb":-16.569775132630912,"num_windows":19,"settings":{"duration":1,"step":1,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260403-210053/sliding_window_cache.npz","sliding_loaded_from_cache":false}
```

RECAST+conv_mlp: checkpoints/rtpp_20260403-210825 checkpoints/rtpp_20260403-112025

```python
{"nll_train_time": -5.125350475311279, "nll_train_total": -5.172115325927734, "nll_train_bg": -0.04676487669348717, "nll_val_time": -5.8281121253967285, "nll_val_total": -5.853720188140869, "nll_val_bg": -0.025608005002141, "nll_test_time": -4.7189764976501465, "nll_test_total": -4.719761848449707, "nll_test_bg": -0.0007853982388041914, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

```python
{"nll_train_time": -5.072784900665283, "nll_train_total": -5.117547512054443, "nll_train_bg": -0.044762562960386276, "nll_val_time": -5.778948783874512, "nll_val_total": -5.815385341644287, "nll_val_bg": -0.03643663972616196, "nll_test_time": -4.724032878875732, "nll_test_total": -4.728349208831787, "nll_test_bg": -0.004316633101552725, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

RECAST+kernel(Gamma): checkpoints/rtpp_kernel_cb_hab4_20260407

```python
{"nll_train_time": -4.9665656089782715, "nll_train_total": -5.189393997192383, "nll_train_bg": -0.2228284329175949, "nll_val_time": -5.728132724761963, "nll_val_total": -5.863674163818359, "nll_val_bg": -0.13554148375988007, "nll_test_time": -4.541913986206055, "nll_test_total": -4.5930047035217285, "nll_test_bg": -0.05109091475605965, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

```python
{"status":"ok","coverage":0.7368421052631579,"mae":67.08636842105264,"rmse":101.43548106970316,"lp_nb":-6.728047518808794,"num_windows":19,"settings":{"duration":1,"step":1,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/rtpp_kernel_cb_hab4_20260407/sliding_window_cache.npz","sliding_loaded_from_cache":false}
```

RECAST+mamba: checkpoints/rtpp_20260403-170646 

```python
{"nll_train_time": -4.897585391998291, "nll_train_total": -5.023638725280762, "nll_train_bg": -0.12605318427085876, "nll_val_time": -5.674637794494629, "nll_val_total": -5.82581090927124, "nll_val_bg": -0.15117309987545013, "nll_test_time": -4.551606178283691, "nll_test_total": -4.584494590759277, "nll_test_bg": -0.03288847580552101, "num_events_train": 2963, "num_events_val": 995, "num_events_test": 724}
```

```python
{"status":"ok","coverage":0.3684210526315789,"mae":115.6852105263158,"rmse":149.7717735217224,"lp_nb":-16.68617605092327,"num_windows":19,"settings":{"duration":1,"step":1,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260403-170646/sliding_window_cache.npz","sliding_loaded_from_cache":true}
```

4. St1-2018(-0.5):

ETAS+P: checkpoints/etas_20260330-163949

```python
{"nll_train_time": -5.54155969619751, "nll_train_total": -5.54155969619751, "nll_val_time": -5.576829433441162, "nll_val_total": -5.576829433441162, "nll_test_time": -4.8479838371276855, "nll_test_total": -4.8479838371276855, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

```python
{"status":"ok","coverage":0.8333333333333334,"mae":179.80225,"rmse":246.2674202464896,"lp_nb":-6.592710105571999,"num_windows":36,"settings":{"duration":2,"step":2,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-163949/sliding_window_cache.npz","sliding_loaded_from_cache":false}
```

ETAS+conv_mlp: checkpoints/etas_20260330-173120

```python
{"nll_train_time": -5.544078826904297, "nll_train_total": -5.544078826904297, "nll_val_time": -5.576200485229492, "nll_val_total": -5.576200485229492, "nll_test_time": -4.846304416656494, "nll_test_total": -4.846304416656494, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

ETAS+kernel(Gamma): checkpoints/etas_20260407-122517

```python
{"nll_train_time": -5.5440263748168945, "nll_train_total": -5.5440263748168945, "nll_val_time": -5.575652599334717, "nll_val_total": -5.575652599334717, "nll_test_time": -4.8469624519348145, "nll_test_total": -4.8469624519348145, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

```python
{"status":"ok","coverage":1,"mae":128.49524999999997,"rmse":194.84513721507423,"lp_nb":-6.3701901165844275,"num_windows":36,"settings":{"duration":2,"step":2,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/etas_20260407-122517/sliding_window_cache.npz","sliding_loaded_from_cache":false}
```

ETAS+mamba: checkpoints/etas_20260330-142641

```python
{"nll_train_time": -5.545408248901367, "nll_train_total": -5.545408248901367, "nll_val_time": -5.577223300933838, "nll_val_total": -5.577223300933838, "nll_test_time": -4.847962856292725, "nll_test_total": -4.847962856292725, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

```python
{"status":"ok","coverage":1,"mae":112.61869444444443,"rmse":154.71048583285776,"lp_nb":-6.202012709931211,"num_windows":36,"settings":{"duration":2,"step":2,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/etas_20260330-142641/sliding_window_cache.npz","sliding_loaded_from_cache":false}
```

ETAS+mamba+gp: checkpoints/etas_20260330-175345 \*

```python
{"nll_train_time": -5.544462203979492, "nll_train_total": -5.539621829986572, "nll_train_bg_kl": 0.004840136505663395, "nll_val_time": -5.576732158660889, "nll_val_total": -5.550709247589111, "nll_val_bg_kl": 0.02602260559797287, "nll_test_time": -4.847546577453613, "nll_test_total": -4.811316967010498, "nll_test_bg_kl": 0.03622935712337494, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

RECAST+P: checkpoints/rtpp_20260403-203753 checkpoints/rtpp_20260403-201221

```python
{"nll_train_time": -5.299367904663086, "nll_train_total": -5.419363975524902, "nll_train_bg": -0.11999645084142685, "nll_val_time": -5.338460445404053, "nll_val_total": -5.460873603820801, "nll_val_bg": -0.12241298705339432, "nll_test_time": -4.5666656494140625, "nll_test_total": -4.630105495452881, "nll_test_bg": -0.06343961507081985, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

```python
{"status":"ok","coverage":0.2222222222222222,"mae":268.98016666666666,"rmse":372.16601866587865,"lp_nb":-29.271350739070456,"num_windows":36,"settings":{"duration":2,"step":2,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260403-203753/sliding_window_cache.npz","sliding_loaded_from_cache":false}
```

```python
{"nll_train_time": -5.180963039398193, "nll_train_total": -5.369428634643555, "nll_train_bg": -0.1884661763906479, "nll_val_time": -5.221585750579834, "nll_val_total": -5.415493965148926, "nll_val_bg": -0.19390784204006195, "nll_test_time": -4.512923717498779, "nll_test_total": -4.617834091186523, "nll_test_bg": -0.10491015017032623, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

RECAST+conv_mlp: checkpoints/rtpp_20260403-203150 checkpoints/rtpp_20260403-201551

```python
{"nll_train_time": -5.283726692199707, "nll_train_total": -5.410412788391113, "nll_train_bg": -0.12668642401695251, "nll_val_time": -5.324419021606445, "nll_val_total": -5.454462051391602, "nll_val_bg": -0.1300428807735443, "nll_test_time": -4.551939010620117, "nll_test_total": -4.618659973144531, "nll_test_bg": -0.06672097742557526, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

```python
{"status":"ok","coverage":0.25,"mae":278.82566666666673,"rmse":388.6756284942354,"lp_nb":-34.490267938944136,"num_windows":36,"settings":{"duration":2,"step":2,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260403-203150/sliding_window_cache.npz","sliding_loaded_from_cache":false}
```

```python
{"nll_train_time": -5.174332618713379, "nll_train_total": -5.364466667175293, "nll_train_bg": -0.1901341825723648, "nll_val_time": -5.212795257568359, "nll_val_total": -5.413140296936035, "nll_val_bg": -0.2003452479839325, "nll_test_time": -4.506704330444336, "nll_test_total": -4.614500999450684, "nll_test_bg": -0.10779624432325363, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

RECAST+kernel(Gamma): checkpoints/rtpp_20260407-130706

```python
{"nll_train_time": -5.191549301147461, "nll_train_total": -5.444743633270264, "nll_train_bg": -0.2531944215297699, "nll_val_time": -5.264633655548096, "nll_val_total": -5.497406482696533, "nll_val_bg": -0.23277229070663452, "nll_test_time": -4.544147491455078, "nll_test_total": -4.660856246948242, "nll_test_bg": -0.11670882254838943, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

```python
{"status":"ok","coverage":0.3611111111111111,"mae":196.48472222222222,"rmse":277.33922364347717,"lp_nb":-24.98232029474184,"num_windows":36,"settings":{"duration":2,"step":2,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260407-130706/sliding_window_cache.npz","sliding_loaded_from_cache":false}
```

RECAST+mamba: checkpoints/rtpp_20260408-205025 checkpoints/rtpp_20260403-202405

```python
{"nll_train_time": -5.222591876983643, "nll_train_total": -5.434678554534912, "nll_train_bg": -0.21208709478378296, "nll_val_time": -5.2782769203186035, "nll_val_total": -5.450173854827881, "nll_val_bg": -0.1718968003988266, "nll_test_time": -4.533393383026123, "nll_test_total": -4.618219375610352, "nll_test_bg": -0.08482575416564941, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```

```python
{"status":"ok","coverage":0.2777777777777778,"mae":183.1134166666667,"rmse":216.85159264709893,"lp_nb":-21.618232708799887,"num_windows":36,"settings":{"duration":2,"step":2,"quantiles":[2.5,97.5],"samples_per_batch":1000,"view_mode":"zoomed","cache_filename":"sliding_window_cache.npz","load_sliding_cache":true,"force_recompute_sliding":false},"sliding_cache_path":"/root/autodl-tmp/em_eqf/checkpoints/rtpp_20260408-205025/sliding_window_cache.npz","sliding_loaded_from_cache":true}
```

```python
{"nll_train_time": -5.183485984802246, "nll_train_total": -5.4211201667785645, "nll_train_bg": -0.23763450980186462, "nll_val_time": -5.235789775848389, "nll_val_total": -5.436279296875, "nll_val_bg": -0.2004896104335785, "nll_test_time": -4.533199787139893, "nll_test_total": -4.632943153381348, "nll_test_bg": -0.09974372386932373, "num_events_train": 22661, "num_events_val": 4211, "num_events_test": 3380}
```
