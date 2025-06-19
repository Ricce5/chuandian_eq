import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool
from torch.nn import Linear
from typing import List
from .SubLayers import CNNBlock
from typing import List, Type




