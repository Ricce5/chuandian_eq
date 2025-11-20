import   torch
import  torch.nn as nn
import src.data
from typing import Dict, Tuple, List, Optional
from src.utils.mask_utils import  get_non_pad_mask

class SM_T_InputAdapter:
    def __call__(self, bx):
        return {
            "event_mark": torch.cat((bx[:, :, 3:5], bx[:, :, 2:3]), dim=-1),  
            "event_time": bx[:, :, 1] 
        }


class S_T_M_InputAdapter:
    def __call__(self, bx):
        return {
            "event_loc": bx[:, :, 3:5],     
            "event_mag": bx[:, :, 2:3],       
            "event_time": bx[:, :, 1],     
        }

    
class SM_T_InputAdapterWithTime:
    def __call__(self, bx):
        return {
            "event_mark": torch.cat((bx[:, :, 3:5], bx[:, :, 2:3]), dim=-1),
            "event_time": bx[:, :, 1]
        }

    def get_extra_inputs(self, bx):
        return {
            "event_time": bx[:, :, 1]  
        }



class SM_T_BatchInputAdapter:
    def __call__(self, batch):
        return {
            "event_mark": torch.cat([batch.loc,batch.mag[...,None]],dim=-1),
            "event_time":  batch.arrival_times
        }

class Type_T_BatchInputAdapter:
    def __call__(self, batch):
        return {
            "event_type": batch.type_seq,
            "event_time":  batch.arrival_times
        }
        
class M_T_InputAdapter:
    def __call__(self, bx):
        return {
            "event_mark": bx[:, :, 2:3],       
            "event_time": bx[:, :, 1],     
        }
    
class M_T_InputAdapterWithTime:
    def __call__(self, bx):
        return {
            "event_mark": bx[:, :, 2:3],       
            "event_time": bx[:, :, 1],     
        }

    def get_extra_inputs(self, bx):
        return {
            "event_time": bx[:, :, 1]  
        }
    
class THP_BatchInputAdapter:
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model 

    def __call__(self, bx: src.data.Batch) -> dict:
        log_inter_times = torch.log(torch.clamp_min(bx.inter_times, 1e-10)).unsqueeze(-1)
        log_inter_times -= self.model.log_tau_mean 

        mag = bx.mag[:, :, None] - self.model.mag_mean 
        # mark = torch.cat([log_inter_times, mag], dim=-1)
        mark = mag 

        return {
            "event_mark": mark * bx.input_mask[:, :, None],
            # "event_time": bx.arrival_times/self.model.time_max * bx.input_mask,  
            "event_time": bx.arrival_times * bx.input_mask, 
            "input_mask": bx.input_mask.float(),
        }
    
class THP_Logdeltat_BatchInputAdapter:
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model

    def __call__(self, bx: src.data.Batch) -> dict:
        log_inter_times = torch.log(torch.clamp_min(bx.inter_times, 1e-10)).unsqueeze(-1)
        log_inter_times -= self.model.log_tau_mean

        mag = bx.mag[:, :, None] - self.model.mag_mean 
        # mark = torch.cat([log_inter_times, mag], dim=-1)
        mark = mag 

        return {
            "event_mark": mark * bx.input_mask[:, :, None],
            # "event_time": bx.arrival_times/self.model.time_max * bx.input_mask,  
            "event_time": bx.arrival_times * bx.input_mask, 
            "log_inter_time": log_inter_times * bx.input_mask[:, :, None],
            "input_mask": bx.input_mask.float(),
        }

class Mixer_BatchInputAdapter:
    def __init__(self, args):
        """
        input adapter for mixer_tpp
        """
        to_t = lambda x: torch.tensor(x, dtype=torch.float32)

        # ---- stats & constants
        self.tau_mean           = to_t(args.get('tau_mean'))
        self.tau_min            = to_t(args.get('tau_min'))
        self.tau_max            = to_t(args.get('tau_max'))
        self.tau_q05           = to_t(args.get('tau_q05'))
        self.tau_q025          = to_t(args.get('tau_q025'))
        self.log_tau_mean       = self.tau_mean.log()
        self.mag_mean           = to_t(args.get('mag_mean'))
        self.time_max           = to_t(args.get('time_max'))
        self.richter_b          = to_t(args.get('richter_b_mle'))
        self.mag_completeness   = to_t(args.get('mag_completeness'))
        self.eps: float          = 1e-10

        # ---- config
        self.extra_input_keys: List[str]    = getattr(args, 'extra_input_keys', ['inter_times', 'times'])
        self.features_input_keys: List[str] = sorted(getattr(args, 'features_input_keys', ['log_inter_times', 'mag']))
        self.normalize_time: bool           = getattr(args, 'normalize_time_by_token', False)

        self.time_scale_base = torch.tensor(1.0, dtype=torch.float32)
        if self.normalize_time:
            if getattr(args, 'time_scale_base', None) is not None:
                base = args.time_scale_base
                print(f"using time scale base {base} (from args.time_scale_base)")
            else:
                key = getattr(args, 'time_scale_base_key', 'tau_mean')
                base = args.get(key, 1.0)
                print(f"using time scale base {base} (key: {key})")
            self.time_scale_base = to_t(base)
            print(f"tau mean: {self.tau_mean}")


    # =========================
    # PIPELINE
    # =========================
    def __call__(self, batch: 'src.data.Batch') -> Dict[str, torch.Tensor]:
        arrival_times, inter_times, mag, loc = self._extract_fields(batch)
        if not hasattr(batch, 'input_mask'):
            input_mask = self._make_non_pad_mask(inter_times).squeeze(-1).float()
        else:
            input_mask = batch.input_mask.float()
        features = self._build_features(mag, loc, inter_times, input_mask.unsqueeze(-1))
        extras = self._build_extras(arrival_times, inter_times, input_mask.unsqueeze(-1))
        return {"features": features, "input_mask": input_mask, **extras}

    # ---------- step 1: parse ----------
    def _extract_fields(self, batch: 'src.data.Batch') -> Tuple[torch.Tensor, ...]:
        """Extract relevant fields from the batch object."""
        arrival_times = batch.arrival_times
        inter_times = batch.inter_times
        mag = batch.mag
        loc = getattr(batch, 'loc', None) # loc might be optional
        return arrival_times, inter_times, mag, loc

    # ---------- step 2: mask ----------
    def _make_non_pad_mask(self, inter_times: torch.Tensor) -> torch.Tensor:
        return get_non_pad_mask(inter_times)  

    # ---------- step 3: features ----------
    def _build_features(
        self,
        mag: torch.Tensor,
        loc: torch.Tensor,
        inter_times: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        feature_map = {
            "log_inter_times": self.normalize_log_inter_times(inter_times),
            "mag": self.normalize_magnitude(mag),
            "loc": loc.unsqueeze(-1) if loc is not None and "loc" in self.features_input_keys else None
        }
        parts = []
        for key in self.features_input_keys:
            if key in feature_map and feature_map[key] is not None:
                parts.append(feature_map[key])
            else:
                raise ValueError(f"Unsupported or missing feature key: {key}")
        
        features = torch.cat(parts, dim=-1)
        return features * non_pad_mask # Apply mask at the end

    # ---------- step 4: extras ----------
    def _build_extras(
        self,
        arrival_times: torch.Tensor,
        inter_times: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        
        if "times" in self.extra_input_keys:
            out["times"] = self.normalize_arrival_times(arrival_times, self.normalize_time) * non_pad_mask.squeeze(-1)
        if "inter_times" in self.extra_input_keys:
            out["inter_times"] = self.normalize_inter_times(inter_times) * non_pad_mask.squeeze(-1)
        return out

    # =========================
    # NORMALIZERS
    # =========================
    def normalize_log_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        log_tau = torch.log(torch.clamp_min(inter_times, self.eps)).unsqueeze(-1)
        return log_tau - self.log_tau_mean.to(log_tau.device)

    def normalize_magnitude(self, mag: torch.Tensor) -> torch.Tensor:
        return mag.unsqueeze(-1) - self.mag_mean.to(mag.device)

    def normalize_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        return (inter_times - self.tau_min.to(inter_times.device)) / (self.tau_max.to(inter_times.device) - self.tau_min.to(inter_times.device) + self.eps)

    def normalize_arrival_times(self, arrival_times: torch.Tensor, normalize_time: bool = False) -> torch.Tensor:
        device = arrival_times.device
        if normalize_time:
            return arrival_times / self.time_scale_base.to(device) 
        else:
            return arrival_times 




class MixerInputAdapterWithTime:
    """
    input adapter of mixer for classification and regression tasks
    """
    def __init__(self, args):
        stats = args.stats
        to_t = lambda x: torch.tensor(x, dtype=torch.float32)

        # ---- stats & constants
        self.tau_mean        = to_t(stats['tau_mean'])
        self.tau_min         = to_t(stats['tau_min'])
        self.tau_max         = to_t(stats['tau_max'])
        self.tau_q025        = to_t(stats['tau_q025'])
        self.tau_q05         = to_t(stats['tau_q05'])
        self.tau_unfiltered  = to_t(stats['tau_unfiltered'])
        self.log_tau_mean    = self.tau_mean.log()
        self.eps: float      = 1e-10

        # ---- config
        self.extra_input_keys: List[str]    = getattr(args, 'extra_input_keys', ['inter_times', 'times'])
        self.features_input_keys: List[str] = sorted(getattr(args, 'features_input_keys', ['mag']))
        self.Twindow: float                 = getattr(args, 'Twindow', None)
        self.normalize_time: bool           = getattr(args, 'normalize_time_by_token', False)

        # ---- time-scale base
        self.time_scale_base = torch.tensor(1.0, dtype=torch.float32)
        if self.normalize_time:
            if getattr(args, 'time_scale_base', None) is not None:
                base = args.time_scale_base
                print(f"using time scale base {base} (from args.time_scale_base)")
            else:
                key = getattr(args, 'time_scale_base_key', 'tau_unfiltered')
                base = stats.get(key, 1.0)
                print(f"using time scale base {base} (key: {key})")
            self.time_scale_base = to_t(base)
            print(f"tau mean: {self.tau_mean}")

    # =========================
    # PIPELINE
    # =========================
    def __call__(self, batch_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        arrival_times, arrival_times_nl, mag, loc, inter_times = self._extract_fields(batch_tensor)
        non_pad_mask = self._make_non_pad_mask(arrival_times)
        features = self._build_features(mag, loc, inter_times, non_pad_mask)
        extras = self._build_extras(arrival_times_nl, inter_times, non_pad_mask)
        return {"features": features, "non_pad_mask": non_pad_mask, **extras}

    # ---------- step 1: parse ----------
    def _extract_fields(self, batch_tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Expected input format:
        [:, :, 0] = arrival_times
        [:, :, 1] = arrival_times_nl
        [:, :, 2:3] = mag
        [:, :, 3:5] = loc
        [:, :, -1]  = inter_times
        """
        arrival_times     = batch_tensor[:, :, 0]
        arrival_times_nl  = batch_tensor[:, :, 1]
        mag               = batch_tensor[:, :, 2:3]
        loc               = batch_tensor[:, :, 3:5]
        inter_times       = batch_tensor[:, :, -1]
        return arrival_times, arrival_times_nl, mag, loc, inter_times

    # ---------- step 2: mask ----------
    def _make_non_pad_mask(self, arrival_times: torch.Tensor) -> torch.Tensor:
        return get_non_pad_mask(arrival_times)  # [B, T, 1]

    # ---------- step 3: features ----------
    def _build_features(
        self,
        mag: torch.Tensor,
        loc: torch.Tensor,
        inter_times: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        feature_map = {
            "mag": mag,                                           # [B, T, 1]
            "log_inter_times": self.normalize_log_inter_times(inter_times),  # [B, T, 1]
            "loc": loc                                            # [B, T, 2]
        }
        parts = [feature_map[k] for k in self.features_input_keys]
        features = torch.cat(parts, dim=-1) * non_pad_mask
        return features  # [B, T, D]

    # ---------- step 4: extras ----------
    def _build_extras(
        self,
        arrival_times_nl: torch.Tensor,
        inter_times: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if "times" in self.extra_input_keys:
            times = self.normalize_arrival_times_nl(arrival_times_nl) * non_pad_mask.squeeze(-1)
            out["times"] = times  # [B, T]

        if "inter_times" in self.extra_input_keys:
            inter = self.normalize_inter_times(inter_times) * non_pad_mask.squeeze(-1)
            out["inter_times"] = inter  # [B, T]

        return out

    # =========================
    # NORMALIZERS
    # =========================
    def normalize_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        return (inter_times - self.tau_min) / (self.tau_max - self.tau_min + self.eps)

    def normalize_log_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        log_tau = torch.log(torch.clamp_min(inter_times, self.eps)).unsqueeze(-1)
        return log_tau - self.log_tau_mean

    def normalize_arrival_times_nl(self, arrival_times_nl: torch.Tensor) -> torch.Tensor:
        """normalize arrival times"""
        scale = self.time_scale_base if self.normalize_time else torch.tensor(1.0, dtype=torch.float32)
        return arrival_times_nl * self.Twindow / scale


    def get_extra_inputs(self, batch_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"event_time": batch_tensor[:, :, 1]}
