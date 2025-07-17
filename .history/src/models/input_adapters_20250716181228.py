import   torch
import  torch.nn as nn
import src.data
from typing import Optional

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
            "event_time": bx.arrival_times/self.model.time_max * bx.input_mask,  # /20000
            "input_mask": bx.input_mask.float(),
        }