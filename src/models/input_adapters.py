import   torch
import  torch.nn as nn
import src.data
from typing import Optional
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
        self.model = model  # 引用 THP 模型实例

    def __call__(self, bx: src.data.Batch) -> dict:
        log_inter_times = torch.log(torch.clamp_min(bx.inter_times, 1e-10)).unsqueeze(-1)
        log_inter_times -= self.model.log_tau_mean  # 使用 buffer

        mag = bx.mag[:, :, None] - self.model.mag_mean  # 使用 buffer
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
        self.model = model  # 引用 THP 模型实例

    def __call__(self, bx: src.data.Batch) -> dict:
        log_inter_times = torch.log(torch.clamp_min(bx.inter_times, 1e-10)).unsqueeze(-1)
        log_inter_times -= self.model.log_tau_mean  # 使用 buffer

        mag = bx.mag[:, :, None] - self.model.mag_mean  # 使用 buffer
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
        self.tau_mean = torch.tensor(args.tau_mean, dtype=torch.float32)
        self.tau_min = torch.tensor(args.tau_min, dtype=torch.float32)
        self.tau_max = torch.tensor(args.tau_max, dtype=torch.float32)
        self.log_tau_mean = self.tau_mean.log()
        self.mag_mean = torch.tensor(args.mag_mean, dtype=torch.float32)
        self.time_max = torch.tensor(args.time_max, dtype=torch.float32)
        self.richter_b = torch.tensor(args.richter_b_mle, dtype=torch.float32)
        self.mag_completeness = torch.tensor(args.mag_completeness, dtype=torch.float32)
        self.extra_input_keys = getattr(args, 'extra_input_keys', ['inter_times', 'times'])
        # Sort feature keys to make order-insensitive
        self.features_input_keys = sorted(getattr(args, 'features_input_keys', ['log_inter_times','mag']))
        self.by_token = getattr(args, 'normalize_time_by_token', False)

    def __call__(
        self,
        batch: src.data.Batch,
    ) -> dict:

        features_parts = []
        for key in self.features_input_keys:
            if key == "log_inter_times":
                log_inter_times = self.normalize_log_inter_times(batch.inter_times)
                features_parts.append(log_inter_times)
            elif key == "mag":
                mag = self.normalize_magnitude(batch.mag)
                features_parts.append(mag)
            elif key == "loc":
                features_parts.append(batch.loc.unsqueeze(-1))
            else:
                raise ValueError(f"Unsupported feature key: {key}")
        features = torch.cat(features_parts, dim=-1).contiguous()
       
        output = {
            "features": features * batch.input_mask[:, :, None],
            "input_mask": batch.input_mask.float(),
        }
        if "times" in self.extra_input_keys:
            output["times"] = self.normalize_arrival_times(batch.arrival_times, self.by_token) * batch.input_mask
        if "inter_times" in self.extra_input_keys:
            output["inter_times"] = self.normalize_inter_times(batch.inter_times) * batch.input_mask
        return output
    
    def normalize_log_inter_times(self, inter_times): 
        log_tau = torch.log(torch.clamp_min(inter_times, 1e-10)).unsqueeze(-1)
        return log_tau - self.log_tau_mean
    
    def normalize_magnitude(self, mag):
        return mag.unsqueeze(-1) - self.mag_mean
    
    def normalize_inter_times(self, inter_times):
        return (inter_times - self.tau_min) / (self.tau_max - self.tau_min + 1e-10)
      

    def normalize_arrival_times(self, arrival_times, by_token=False): 
        if by_token:
            return (arrival_times - arrival_times[:, 0:1]) / self.tau_mean
        else:
            return arrival_times - arrival_times[:, 0:1]




class MixerInputAdapterWithTime:
    def __init__(self, args):
        self.tau_mean = torch.tensor(args.stats['tau_mean'], dtype=torch.float32)
        self.tau_min = torch.tensor(args.stats['tau_min'], dtype=torch.float32)
        self.tau_max = torch.tensor(args.stats['tau_max'], dtype=torch.float32)
        self.tau_q025 = torch.tensor(args.stats['tau_q025'], dtype=torch.float32)
        self.tau_q05 = torch.tensor(args.stats['tau_q05'], dtype=torch.float32)
        self.tau_unfiltered = torch.tensor(args.stats['tau_unfiltered'], dtype=torch.float32)
        self.log_tau_mean = self.tau_mean.log()
        self.eps = 1e-10
        self.extra_input_keys = getattr(args, 'extra_input_keys', ['inter_times', 'times'])
        self.features_input_keys = sorted(getattr(args, 'features_input_keys', ['mag']))
        self.Twindow = getattr(args, 'Twindow', None)  
        self.normalize_time = getattr(args, 'normalize_time_by_token', False)
        # 兼容直接传入 time_scale_base 或通过 key 查找
        if self.normalize_time:
            if hasattr(args, 'time_scale_base') and args.time_scale_base is not None:
                self.time_scale_base = torch.tensor(args.time_scale_base, dtype=torch.float32)
                print(f"using time scale base {self.time_scale_base} (from args.time_scale_base)")
            else:
                key = getattr(args, 'time_scale_base_key', 'tau_unfiltered')
                self.time_scale_base = torch.tensor(
                args.stats.get(key, 1.0),
                dtype=torch.float32
                )
                print(f"using time scale base {self.time_scale_base} (key: {key})")
    def __call__(self, batch_tensor):
        arrival_times = batch_tensor[:, :, 0]
        arrival_times_nl = batch_tensor[:,:,1]
        mag = batch_tensor[:, :, 2:3]
        inter_times = batch_tensor[:, :, -1]
        loc = batch_tensor[:, :, 3:5]

        non_pad_mask = get_non_pad_mask(arrival_times)
        log_inter_times = self.normalize_log_inter_times(inter_times)

        # 动态拼接 features
        features_parts = []
        for key in self.features_input_keys:
            if key == "mag":
                features_parts.append(mag )
            elif key == "log_inter_times":
                features_parts.append(log_inter_times)
            elif key == "loc":
                features_parts.append(loc)
            else:
                raise ValueError(f"Unsupported feature key: {key}")

        features = torch.cat(features_parts, dim=-1)* non_pad_mask

        output = {
            "features": features,
            "non_pad_mask": non_pad_mask
        }

        if "times" in self.extra_input_keys:
            output["times"] = self.normalize_arrival_times_nl(arrival_times_nl, self.normalize_time) * non_pad_mask.squeeze(-1)
        if "inter_times" in self.extra_input_keys:
            output["inter_times"] = self.normalize_inter_times(inter_times) * non_pad_mask.squeeze(-1)

        return output

    def get_extra_inputs(self, batch_tensor):
        return {
            "event_time": batch_tensor[:, :, 1]
        }

    def normalize_inter_times(self, inter_times):
        return (inter_times - self.tau_min) / (self.tau_max - self.tau_min + self.eps)

    def normalize_log_inter_times(self, inter_times):
        log_tau = torch.log(torch.clamp_min(inter_times, self.eps)).unsqueeze(-1)
        return log_tau - self.log_tau_mean

    def normalize_arrival_times_nl(self, arrival_times_nl, by_token = True):
        if by_token:
            return arrival_times_nl * self.Twindow / self.time_scale_base
        else:
            return arrival_times_nl * self.Twindow
