1. CLF:

dataset 	Twindow	Tfore	dt	Mf	context_len	rf val rf test	p/n	
ChuanDian	180	10	   10	4  	1	      0.61	0.67	  1          
    checkpoints/rf_dfae9b13  checkpoints/clf_mixer_attnpl_t_20260205-222827
ChuanDian	200	20	   10	4.5	1	      0.64	0.8	  1	
ChuanDian	180	20	   10	4.5	1	      0.61	0.85	  1          
    checkpoints/rf_847e64da  checkpoints/clf_mixer_attnpl_t_20260205-223841
ChuanDian	180	30	   10	4.5	1	      0.65	0.85	  1.5	~      
    checkpoints/rf_ba2359e6  
ChuanDian	180	30	   10	5	   1	      0.63	0.8	  0.5	
ChuanDian	180	60	   10	5	   1	      0.88	0.86	  1	~*     
    checkpoints/rf_9ffe46be checkpoints/clf_mixer_attnpl_t_20260205-225249
ChuanDian	180	60	   10	5.5	1	      0.8	0.69	  0.25	    
ChuanDian	180	60	   10	5.5	2	      0.95	0.67	  0.25	
ChuanDian	180	90	   10	5.5	1	      0.88	0.67	  0.5	~      
    checkpoints/rf_af684ff4  checkpoints/clf_mixer_attnpl_t_20260205-230257
ChuanDian	180	90	   10	5.5	2	      0.96	0.9	  1	


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
tpp:
tpp+b: checkpoints/mixer_tpp_20260204-225304
tpp+b+pretrian: 
tpp+log:  
tpp+log+b: 
tpp-txpos:
 





SCEDC b值预测 ：checkpoints/mixer_tpp_20260205-111733