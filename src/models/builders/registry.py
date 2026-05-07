from src.utils.registrable import Registrable

class ModelBuilder(Registrable):
    def __call__(self, args, device):
        raise NotImplementedError

__all__ = ["ModelBuilder"]
