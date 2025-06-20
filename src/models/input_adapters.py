import   torch

class SM_T_InputAdapter:
     def __call__(self, bx):
        t_n_seq = bx[:, :, 1]        
        mag_seq = bx[:, :, 2:3]       
        loc_seq = bx[:, :, 3:5]       
        f_seq = torch.cat((loc_seq, mag_seq), dim=-1)  # [B, L, 3]
        return f_seq, t_n_seq
     
class S_M_T_InputAdapter:
    def __call__(self, bx):        
        t_n_seq = bx[:, :, 1]          
        mag_seq = bx[:, :, [2]]         
        loc_seq = bx[:, :, 3:5]         
        return loc_seq, mag_seq, t_n_seq