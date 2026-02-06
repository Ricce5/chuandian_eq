1. CLF:

dataset 	Twindow	Tfore	dt	Mf	context_len 	p/n	
ChuanDian	180	10	   10	4  	1	       1          
    checkpoints/rf_dfae9b13  
    checkpoints/clf_mixer_attnpl_t_20260205-222827
     val: {'precision': 0.7464788732394366, 'recall': 0.9906542056074766, 'f1': 0.8514056224899599, 'auc': 0.8140426551641505, 'pr_auc': 0.9271271534201313, 'fpr': 0.9230769230769231, 'tpr': 0.9906542056074766, 
        'R': 0.0675772825305535, 'conf': 1.8110423215777166e-11, 'threshold': 0.4134446680545807}
      test:  {'precision': 0.6476190476190476, 'recall': 0.8607594936708861, 'f1': 0.7391304347826086, 'auc': 0.7961653015636635, 'pr_auc': 0.8651239527002411, 'fpr': 0.5441176470588235, 'tpr': 0.8607594936708861, 
        'R': 0.3166418466120626, 'conf': 0.9990411429982295, 'threshold': 0.44980189204216003}
ChuanDian	180	20	   10	4.5	1	    	  1          
    checkpoints/rf_847e64da  
    checkpoints/clf_mixer_attnpl_t_20260205-223841
    val: {'precision': 0.5145631067961165, 'recall': 0.8833333333333333, 'f1': 0.6503067484662577, 'auc': 0.7248062015503877, 'pr_auc': 0.6949490218519075, 'fpr': 0.5813953488372093, 'tpr': 0.8833333333333333, 
        'R': 0.30193798449612397, 'conf': 0.726927832278985, 'threshold': 0.3267044425010681}
    test:  {'precision': 0.75, 'recall': 0.6, 'f1': 0.6666666666666666, 'auc': 0.8346396965865992, 'pr_auc': 0.7392454081253295, 'fpr': 0.061946902654867256, 'tpr': 0.6, 
        'R': 0.5380530973451327, 'conf': 1.0, 'threshold': 0.37116697430610657}
ChuanDian	180	30	   10	4.5	1	    	  1.5	~      
    checkpoints/rf_ba2359e6  
    checkpoints/clf_mixer_attnpl_t_20260205-213430
    val:  {'precision': 0.6377952755905512, 'recall': 0.9759036144578314, 'f1': 0.7714285714285715, 'auc': 0.7600401606425703, 'pr_auc': 0.8371695266694815, 'fpr': 0.7666666666666667, 'tpr': 0.9759036144578314, 
        'R': 0.20923694779116464, 'conf': 0.0019910004391618425, 'threshold': 0.2959364354610443}
    test: {'precision': 0.9090909090909091, 'recall': 0.6382978723404256, 'f1': 0.75, 'auc': 0.8993794326241135, 'pr_auc': 0.8516222215159499, 'fpr': 0.03125, 'tpr': 0.6382978723404256, 
        'R': 0.6070478723404256, 'conf': 1.0, 'threshold': 0.47319287061691284}
ChuanDian	180	60	   10	5	   1	   	  1	~*     
    checkpoints/rf_7310e627
    val:{"precision": 0.7849462365591398, "recall": 0.8390804597701149,"f1": 0.8111111111111111,"auc": 0.8045977011494254,"pr_auc": 0.865765058076529,"fpr": 0.3448275862068966,"tpr": 0.8390804597701149,
        "R": 0.49425287356321834, "conf": 1.0, "threshold": 0.6271477663131956}
    test: { "precision": 0.5180722891566265,"recall": 0.8958333333333334,"f1": 0.6564885496183206,"auc": 0.8229166666666666,"pr_auc": 0.7057947126870211,"fpr": 0.40404040404040403,"tpr": 0.8958333333333334,
        "R": 0.46410642570281124, "conf": 0.9999999999988334, "threshold": 0.5890404340586651}
    checkpoints/clf_mixer_attnpl_t_20260205-225249
    val:  {'precision': 0.6434108527131783, 'recall': 0.9540229885057471, 'f1': 0.7685185185185185, 'auc': 0.7316686484344035, 'pr_auc': 0.8529366516296851, 'fpr': 0.7931034482758621, 'tpr': 0.9540229885057471, 
        'R': 0.16091954022988497, 'conf': 3.86480550837049e-05, 'threshold': 0.2606133818626404}
    test:  {'precision': 0.6666666666666666, 'recall': 0.875, 'f1': 0.7567567567567568, 'auc': 0.8878367003367003, 'pr_auc': 0.849152712364616, 'fpr': 0.21212121212121213, 'tpr': 0.875, 
        'R': 0.6628787878787878, 'conf': 1.0, 'threshold': 0.2804109752178192}
ChuanDian	180	90	   10	5.5	1		  0.5	~      
    checkpoints/rf_af684ff4  
    checkpoints/clf_mixer_attnpl_t_20260206-100920 checkpoints/clf_mixer_attnpl_t_20260205-230257
    val: {'precision': 0.926829268292683, 'recall': 0.6129032258064516, 'f1': 0.7378640776699029, 'auc': 0.8639784946236558, 'pr_auc': 0.8536683537619597, 'fpr': 0.03333333333333333, 'tpr': 0.6129032258064516, 
        'R': 0.5795698924731183, 'conf': 1.0, 'threshold': 0.4257252514362335}
    test: {'precision': 0.2191780821917808, 'recall': 0.8421052631578947, 'f1': 0.34782608695652173, 'auc': 0.7489278752436648, 'pr_auc': 0.27135010976409285, 'fpr': 0.4222222222222222, 'tpr': 0.8421052631578947, 
        'R': 0.41988304093567247, 'conf': 0.999998000327738, 'threshold': 0.3488326668739319}
    


2. CLF Pre-training:
None: checkpoints/clf_mixer_attnpl_t_20260205-220907
AZDX-b：checkpoints/clf_mixer_attnpl_t_20260205-213729 checkpoints/clf_mixer_attnpl_t_20260205-214507
AZDX+b: checkpoints/clf_mixer_attnpl_t_20260205-213430
SCEDC-b: checkpoints/clf_mixer_attnpl_t_20260205-215511
SCEDC+b: checkpoints/clf_mixer_attnpl_t_20260205-215943


3. REG:
lstm    checkpoints/lstm_20250906-152053
reg    checkpoints/reg_mixer_attnpl_t_20260204-180531
+log-pretrain    checkpoints/reg_mixer_attnpl_t_20260204-195612
-txpos   checkpoints/reg_mixer_attnpl_t_20260204-201607
-TMAP   checkpoints/reg_mixer_attnpl_t_20260204-194201 
-pretrain checkpoints/reg_mixer_attnpl_t_20260204-192629


4. TPP
SCEDC
tpp:  checkpoints/mixer_tpp_20260204-233222
tpp+b:  checkpoints/mixer_tpp_20260205-110204 checkpoints/mixer_tpp_20260205-100346
tpp+log: checkpoints/mixer_tpp_20260205-113617
tpp+log+b: checkpoints/mixer_tpp_20260205-115028
tpp-txpos:  checkpoints/mixer_tpp_20260205-120039

ChuanDian
tpp:  checkpoints/mixer_tpp_20260206-102040
tpp+b:  checkpoints/mixer_tpp_20260204-225304
tpp+b+pretrian: 
tpp+log:  checkpoints/mixer_tpp_20260206-102404
tpp+log+b: checkpoints/mixer_tpp_20260206-102755
tpp-txpos:  checkpoints/mixer_tpp_20260206-103125
 



T-XPOS ablation
2*Transformer+T-XPOS:  checkpoints/mixer_tpp_20260206-104534
2*Transformer+T-RoPE:  checkpoints/mixer_tpp_20260206-105017
2*Transformer+XPOS:  checkpoints/mixer_tpp_20260206-105722
2*Transformer+RoPE:  checkpoints/mixer_tpp_20260206-105441



SCEDC 预测，b值绘图 ：checkpoints/mixer_tpp_20260206-112901
    ../checkpoints/mixer_tpp_20260205-111733