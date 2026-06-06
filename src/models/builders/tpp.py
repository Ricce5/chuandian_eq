from .registry import ModelBuilder


def _resolve_loss_weights(args):
    configured = getattr(args, "loss_weights", None)
    if configured is not None:
        return configured

    legacy_weights = {}
    if hasattr(args, "bg_kl_weight"):
        legacy_weights["bg_kl_weight"] = getattr(args, "bg_kl_weight")
    if hasattr(args, "bg_norm_weight"):
        legacy_weights["bg_norm_weight"] = getattr(args, "bg_norm_weight")

    return legacy_weights or None


@ModelBuilder.register("thp")
class THPBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.thp import THP
        from src.models.transformer.transformers import Transformer
        from src.models.input_adapters import THP_BatchInputAdapter
        from src.models.heads import TaskHead
        from src.models.base_model import BaseModel
        import torch.nn as nn
        encoder = Transformer(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn = getattr(args, 'rnn_dropout', 0), 
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
        )
        adapter = THP_BatchInputAdapter(model=None)  
        hypernet_time = nn.Linear(args.d_model, 3 * args.num_components).to(device)
        hypernet_mag = nn.Linear(args.d_model, 1).to(device)
        base_model = BaseModel(encoder=encoder, input_adapter=adapter, device=device)
        return THP(args,base_model, hypernet_time, hypernet_mag)


@ModelBuilder.register("thp_deltat")
class THPDeltatBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.thp import THP
        from src.models.transformer.transformers import Transformer_Logdeltat
        from src.models.input_adapters import THP_Logdeltat_BatchInputAdapter
        from src.models.heads import TaskHead
        from src.models.base_model import BaseModel
        import torch.nn as nn
        encoder = Transformer_Logdeltat(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn = getattr(args, 'rnn_dropout', 0), 
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
        )
        adapter = THP_Logdeltat_BatchInputAdapter(model=None)  
        hypernet_time = nn.Linear(args.d_model+1, 3 * args.num_components).to(device)
        hypernet_mag = nn.Linear(args.d_model+1, 1).to(device)
        base_model = BaseModel(encoder=encoder, input_adapter=adapter, device=device)
        return THP(args,base_model, hypernet_time, hypernet_mag)


@ModelBuilder.register("rtpp")
class RTTPBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.recurrent import RecurrentTPP
        if  getattr(args,'bg_model', None) is not None:
            from src.models.bg import BGModel
            bg_model = BGModel.by_name(args.bg_model)(**args.bg_model_cfg, device=device)
        else:
            bg_model = None
        return RecurrentTPP(args, device, bg_model)


@ModelBuilder.register("rtpp_v2")
class RTTPV2Builder(ModelBuilder):
    def __call__(self, args, device):
        import math
        import torch.nn as nn

        from src.models.tpp.common.recurrent_blocks import RNNTPPBackbone, build_time_hypernet
        from src.models.tpp.recurrent.model_v2 import RecurrentTPPV2

        if getattr(args, "bg_model", None) is not None:
            from src.models.bg import BGModel

            bg_model = BGModel.by_name(args.bg_model)(**args.bg_model_cfg, device=device)
        else:
            bg_model = None

        time_preprocess = str(getattr(args, "rtpp_time_preprocess", "legacy")).strip().lower()
        tau_mean_value = max(float(args.tau_mean), 1e-10)

        def _resolve_log_tau_std(*, default_value: float, convert_ln_to_log10: bool) -> float:
            if not hasattr(args, "log_tau_std"):
                return float(getattr(args, "rtpp_log_tau_std", default_value))

            raw_log_tau_std = float(args.log_tau_std)
            if not math.isfinite(raw_log_tau_std):
                raise ValueError(f"args.log_tau_std must be finite, got {raw_log_tau_std}.")
            if raw_log_tau_std <= 0:
                raise ValueError(f"args.log_tau_std must be > 0, got {raw_log_tau_std}.")
            if convert_ln_to_log10:
                return max(raw_log_tau_std / math.log(10.0), 1e-8)
            return max(raw_log_tau_std, 1e-8)

        if time_preprocess == "oracle":
            log_tau_mean = math.log10(tau_mean_value)
            log_tau_std = _resolve_log_tau_std(default_value=2.0, convert_ln_to_log10=True)
        else:
            log_tau_mean = math.log(tau_mean_value)
            log_tau_std = _resolve_log_tau_std(default_value=1.0, convert_ln_to_log10=False)

        backbone = RNNTPPBackbone(
            context_size=args.d_model,
            tau_mean=args.tau_mean,
            mag_mean=args.mag_mean,
            rnn_type=args.rnn_type,
            num_rnn_layers=getattr(args, "num_rnn_layers", 1),
            dropout=getattr(args, "rnn_dropout", 0.0),
            input_magnitude=True,
            num_extra_features=None,
            use_residual=getattr(args, "rnn_use_residual", False),
            use_layernorm=getattr(args, "rnn_use_layernorm", False),
            time_preprocess=time_preprocess,
            log_tau_mean=log_tau_mean,
            log_tau_std=log_tau_std,
            inter_time_min=getattr(args, "rtpp_inter_time_min", 1e-10),
            inter_time_max=getattr(args, "rtpp_inter_time_max", 1e10),
        )
        num_time_params = 3 * args.num_components
        hypernet_time = build_time_hypernet(
            input_dim=args.d_model,
            output_dim=num_time_params,
            use_mlp=getattr(args, "hypernet_time_use_mlp", True),
            hidden_dim=getattr(args, "hypernet_time_hidden_dim", args.d_model),
            activation=getattr(args, "hypernet_time_activation", "silu"),
            dropout=getattr(args, "hypernet_time_mlp_dropout", 0.0),
        ).to(device)
        hypernet_mag = nn.Linear(args.d_model, 1).to(device)
        return RecurrentTPPV2(
            args,
            backbone=backbone,
            hypernet_time=hypernet_time,
            hypernet_mag=hypernet_mag,
            device=device,
            bg_model=bg_model,
        )


@ModelBuilder.register("oracle")
class OracleBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.oracle import Oracle

        return Oracle(args, device)


@ModelBuilder.register("nhpp")
class NHPPBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.nhpp import NHPP
        from src.models.bg import BGModel
        assert hasattr(args, 'bg_model'), "NHPP requires a background model."
        bg_model = BGModel.by_name(args.bg_model)(**args.bg_model_cfg, device=device)
        return NHPP(args, device, bg_model)


@ModelBuilder.register("etas")
class ETASBuilder(ModelBuilder):
    def __call__(self, args, device):
        import torch
        from src.models.tpp.etas import ETAS

        richter_b = args.richter_b_mle
    
        mag_completeness = args.mag_completeness
        mag_max = getattr(args, 'mag_max', 10)
        if getattr(args, "bg_model", None) is not None:
            base_rate_default = 0.0
        else:
            base_rate_default = 0.26
        base_rate_init = torch.tensor(
            getattr(args, "base_rate_init", base_rate_default),
            dtype=torch.float64,
        )
        omori_p_init = float(getattr(args, "omori_p_init", 1.08))
        omori_c_init = float(getattr(args, "omori_c_init", 0.1))
        productivity_k_init = float(getattr(args, "productivity_k_init", 0.0073))
        productivity_alpha_init = float(getattr(args, "productivity_alpha_init", 1.0))

        if getattr(args, 'bg_model', None) is not None:
            from src.models.bg import BGModel
            bg_model = BGModel.by_name(args.bg_model)(**args.bg_model_cfg, device=device)
        else:
            bg_model = None

        model = ETAS(
            omori_p_init=omori_p_init,
            omori_c_init=omori_c_init,
            base_rate_init=base_rate_init,
            productivity_k_init=productivity_k_init,
            productivity_alpha_init=productivity_alpha_init,
            richter_b=richter_b,
            mag_completeness=mag_completeness,
            mag_max=mag_max,
            device=device,
            bg_model=bg_model,
            fix_mu=getattr(args, "fix_mu", False),
            fixed_mu_value=getattr(args, "fixed_mu_value", None),
            loss_reduction=getattr(args, "loss_reduction", "per_time"),
            query_chunk_size=getattr(args, "etas_query_chunk_size", 0),
            history_chunk_size=getattr(args, "etas_history_chunk_size", 0),
            use_grad_checkpoint=getattr(args, "etas_grad_checkpoint", False),
            enforce_subcritical=getattr(args, "etas_enforce_subcritical", True),
            max_branching_ratio=getattr(args, "etas_max_branching_ratio", 0.98),
            effective_branching_t_max=getattr(args, "etas_effective_branching_t_max", 1e4),
            enforce_p_gt_one=getattr(args, "etas_enforce_p_gt_one", True),
            min_omori_p=getattr(args, "etas_min_omori_p", 1.001),
            constraint_softness=getattr(args, "etas_constraint_softness", 1e-3),
            loss_weights=_resolve_loss_weights(args),
        )
        if getattr(args, "use_double_precision", False):
            model.double()
        else:
            model.float()
        if model.bg_model is not None:
            model.bg_model.float()

        return model


@ModelBuilder.register("etas_zhuang")
class ETASZhuangBuilder(ModelBuilder):
    def __call__(self, args, device):
        import torch
        from src.models.tpp.etas_zhuang import ETASZhuang

        richter_b = args.richter_b_mle
        mag_completeness = args.mag_completeness
        mag_max = getattr(args, "mag_max", 10)

        if getattr(args, "bg_model", None) is not None:
            from src.models.bg import BGModel

            bg_model = BGModel.by_name(args.bg_model)(**args.bg_model_cfg, device=device)
            base_rate_init = torch.tensor(getattr(args, "base_rate_init", 0.0), dtype=torch.float64)
        else:
            bg_model = None
            base_rate_init = torch.tensor(0.26, dtype=torch.float64)

        model = ETASZhuang(
            base_rate_init=base_rate_init,
            productivity_K_init=getattr(args, "productivity_K_init", 0.11),
            productivity_alpha_e_init=getattr(args, "productivity_alpha_e_init", 2.302585092994046),
            richter_b=richter_b,
            mag_completeness=mag_completeness,
            mag_max=mag_max,
            device=device,
            bg_model=bg_model,
            fix_mu=getattr(args, "fix_mu", False),
            fixed_mu_value=getattr(args, "fixed_mu_value", None),
            loss_reduction=getattr(args, "loss_reduction", "per_time"),
            query_chunk_size=getattr(args, "etas_query_chunk_size", 0),
            history_chunk_size=getattr(args, "etas_history_chunk_size", 0),
            use_grad_checkpoint=getattr(args, "etas_grad_checkpoint", False),
            enforce_subcritical=getattr(args, "etas_enforce_subcritical", True),
            max_branching_ratio=getattr(args, "etas_max_branching_ratio", 0.98),
            enforce_p_gt_one=getattr(args, "etas_enforce_p_gt_one", True),
            min_omori_p=getattr(args, "etas_min_omori_p", 1.001),
            constraint_softness=getattr(args, "etas_constraint_softness", 1e-3),
            loss_weights=_resolve_loss_weights(args),
        )
        if getattr(args, "use_double_precision", False):
            model.double()
        else:
            model.float()
        if model.bg_model is not None:
            model.bg_model.float()

        return model


@ModelBuilder.register("mtpp")
class MTTPBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.mtpp import MambaTPP
        return MambaTPP(args, device)


@ModelBuilder.register("mhp")
class MHPBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.mhp import MHP
        return MHP(args, device)


@ModelBuilder.register("btpp")
class BTTPBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.btpp import BlockTPP
        return BlockTPP(args, device)


@ModelBuilder.register("njdtpp")
class NJDTPPBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.njdtpp import NJDTPP

        return NJDTPP(args, device)


@ModelBuilder.register("mixer_tpp")
class MixerTPPBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.mixer_tpp import MixerTPP
        from src.models.bg import BGModel
        from src.models.input_adapters import MixerBatchAdapter
        from src.models.heads import TaskHead
        from src.models.base_model import BaseModel
        from src.models.mamba.mixer_seq import MixerModel,MixerModelWrapper
        from src.models.mamba.scan_wrapper import BoundedSelectiveScanWrapper,BoundedDiscreteSSM
        from src.models.layers import Causalconv
        from mamba_ssm import Mamba
        from src.models.mamba.scan_wrapper import  SelectiveScanWrapper
        import torch
        import torch.nn as nn   
        encoder = MixerModel(**args.mixer_model_config, device=device, dtype=torch.float32).to(device)
        adapter = MixerBatchAdapter(args)
        hypernet_time = nn.Linear(args.d_model, 3 * args.num_components).to(device)
        hypernet_mag = nn.Sequential(
            nn.Linear(args.d_model, 1).to(device),
        )
        # Configure optional b_filter
        if getattr(args, 'use_ssm_filter', False):
            fiter_type = getattr(args, 'ssm_filter_type', 'ssm')
            if fiter_type == 'ssm':
                b_filter = SelectiveScanWrapper(
                    d_model=args.d_model,
                    d_state=16,
                    use_D=False,
                    device=device,
                )
            elif fiter_type == 'bounded_ssm':
                b_filter = BoundedSelectiveScanWrapper(
                    d_model=args.d_model,
                    d_state=16,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown b_filter_type: {fiter_type}")
        else:
            b_filter = None

        base_model = MixerModelWrapper(encoder=encoder, input_adapter=adapter, device=device)
        predict_b = getattr(args, 'predict_b', False)
        use_b_updater = getattr(args, 'use_b_updater', False)
        loss_weights = getattr(args, 'loss_weights', None)
        loss_reduction = getattr(args, 'loss_reduction', None)
        use_adaptive_loss_weights = getattr(args, 'use_adaptive_loss_weights', False)
        b_range = getattr(args, 'b_range', None)
        b_init = getattr(args, 'b_init', 1.0)
        if getattr(args, 'bg_model', None) is not None:
            bg_model = BGModel.by_name(args.bg_model)(**args.bg_model_cfg, device=device)
        else:
            bg_model = None
        return MixerTPP(base_model, hypernet_time, hypernet_mag, dropout=args.dropout,
                        predict_b=predict_b, use_b_updater=use_b_updater,
                        loss_weights=loss_weights, loss_reduction=loss_reduction,b_range=b_range,
                        use_adaptive_loss_weights=use_adaptive_loss_weights,
                        bg_model=bg_model,
                        b_filter=b_filter,
                        b_init=b_init)
