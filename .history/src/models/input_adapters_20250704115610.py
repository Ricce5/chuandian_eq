import   torch

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
        
        

class T_M_InputAdapter:
    def __call__(self, bx):
        return {
            "event_mag": bx[:, :, 2:3],       
            "event_time": bx[:, :, 1],     
        }