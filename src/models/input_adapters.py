import   torch

class SM_T_InputAdapter:
    def __call__(self, bx):
        return {
            "event_mark": torch.cat((bx[:, :, 3:5], bx[:, :, 2:3]), dim=-1),  # loc + mag
            "event_time": bx[:, :, 1]  # normalized time
        }


class S_T_M_InputAdapter:
    def __call__(self, bx):
        return {
            "event_loc": bx[:, :, 3:5],       # 纬度，经度
            "event_mag": bx[:, :, 2:3],       # 震级
            "event_time": bx[:, :, 1],        # 归一化时间
        }

    
class SM_T_InputAdapterWithTime:
    def __call__(self, bx):
        return {
            "event_mark": torch.cat((bx[:, :, 3:5], bx[:, :, 2:3]), dim=-1),  # loc + mag
            "event_time": bx[:, :, 1]  # normalized time
        }

    def get_extra_inputs(self, bx):
        return {
            "event_time": bx[:, :, 1]  # 提取时间信息，注意保持和 extractor 期望的 shape 一致
        }



