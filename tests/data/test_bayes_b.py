# %%
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
from src.data.bayesian_b_updater import BayesianGRBUpdater
args= load_args_from_yaml("./config/mixer_tpp.yaml")
# args.dataset = 'SCEDC'
base_dir = f"./data/{args.dataset}"
# %%
seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data_tpp(base_dir=base_dir, args=args,)
# %%
# %%
updater = BayesianGRBUpdater(Mc=4.5, delta=0.99, a0=20, init_b_target=1, mag_key="mag", write_back=True)
results = updater.fit(seq)            # Update events sequentially; also write the result fields back to seq
BayesianGRBUpdater.plot(seq)          # Plot (optional: pass in truth_lines / switch_index)
# Append an event online:
# updater.update_one(3.2)               # Return the summary after this event

# %%
from src.data.batch import Batch
batch = Batch.from_list([seq])   
# %%
from src.distributions.gutenberg_richter import GutenbergRichter
dist = GutenbergRichter(b=batch.b_mean, mag_min=3.0, mag_max=10.0)
dist.log_likelihood(batch.mag,batch.nll_event_mask.bool())/batch.t_end - batch.t_nll_start 
# %%
dist_constant = GutenbergRichter(b= args.richter_b_mle, mag_min=3.0, mag_max=10.0)
dist_constant.log_likelihood(batch.mag,batch.nll_event_mask.bool())/batch.t_end
# %%
from src.distributions import gamma
dist_b = gamma.Gamma(batch.a_t, batch.s_t * torch.log(torch.tensor(10.0, device=batch.device)))
# %%
dist_b.mean
# %%
torch.max(dist_b.mean)
# %%
torch.min(dist_b.mean)
# %%
batch.a_t/(batch.s_t * torch.log(torch.tensor(10.0, device=batch.device)))
# %%
