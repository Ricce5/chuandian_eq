"""Public exports for shared TPP common modules.

`src.models.tpp.common` is the low-coupling foundation layer for reusable TPP
components (distribution decoding, recurrent/oracle blocks, and sequence ops).
"""

from .inter_time_decoding import WeibullMixtureDecoder
from .oracle_blocks import OracleDistDecoder, OracleFCNDecoder, OracleRNNEncoder
from .recurrent_blocks import FiLMContextFuse, RNNTPPBackbone, TimeHypernetMLP, build_time_hypernet
from .etas_utils import (
    gen_magnitude,
    iter_chunks,
    omori_history_contribution,
    omori_integral_np,
    resolve_chunk_size,
    sample_omori_truncated_np,
    sample_omori_truncated_torch,
    soft_lower_bound,
    soft_upper_bound,
    to_tensor_like,
    torch_gen_magnitude,
    torch_omori_integral,
)
from .sequence_ops import build_sample_batch, evaluate_compensator_from_model

__all__ = [
    "WeibullMixtureDecoder",
    "OracleRNNEncoder",
    "OracleFCNDecoder",
    "OracleDistDecoder",
    "FiLMContextFuse",
    "RNNTPPBackbone",
    "TimeHypernetMLP",
    "build_time_hypernet",
    "gen_magnitude",
    "iter_chunks",
    "omori_history_contribution",
    "omori_integral_np",
    "resolve_chunk_size",
    "sample_omori_truncated_np",
    "sample_omori_truncated_torch",
    "soft_lower_bound",
    "soft_upper_bound",
    "to_tensor_like",
    "torch_gen_magnitude",
    "torch_omori_integral",
    "build_sample_batch",
    "evaluate_compensator_from_model",
]
