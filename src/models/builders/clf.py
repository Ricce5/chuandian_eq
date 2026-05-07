from .registry import ModelBuilder


@ModelBuilder.register("classifier")
class ClassifierBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer_ST
        from src.models.input_adapters import SM_T_InputAdapter
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.extractors import RepresentationExtractor
        from src.models.base_model import BaseModel

        encoder = Transformer_ST(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
        )

        input_adapter = SM_T_InputAdapter()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)

        extractor = RepresentationExtractor.by_name("last")()

        head = TaskHead(
            input_dim=3 * args.d_model,  
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None 
        )


@ModelBuilder.register("classifier_tm_s")
class ClassifierTMSBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer
        from src.models.input_adapters import M_T_InputAdapter
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.extractors import RepresentationExtractor
        from src.models.base_model import BaseModel

        encoder = Transformer(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
        )

        input_adapter = M_T_InputAdapter()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)

        extractor = RepresentationExtractor.by_name("last")()

        head = TaskHead(
            input_dim= args.d_model,  
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None 
        )


@ModelBuilder.register("clf_tm_attnpl")
class ClassifierTMSAttnPlBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer
        from src.models.input_adapters import M_T_InputAdapter
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.extractors import RepresentationExtractor
        from src.models.base_model import BaseModel


        encoder = Transformer(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
        )

        input_adapter = M_T_InputAdapter()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)

        extractor = RepresentationExtractor.by_name("attn")(  
            input_dim= args.d_model,
            hidden_dim= 2*args.d_model,
            device=device)

       
        head = TaskHead(
            input_dim= args.d_model,  
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None 
        )


@ModelBuilder.register("clf_tm_attnpl_t")
class ClassifierTMSAttnPlTBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer
        from src.models.input_adapters import M_T_InputAdapterWithTime
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.extractors import RepresentationExtractor
        from src.models.base_model import BaseModel

        
        encoder = Transformer(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
        )

        input_adapter = M_T_InputAdapterWithTime()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)

        extractor = RepresentationExtractor.by_name("attn_time")(  
            input_dim= args.d_model+1,
            hidden_dim= 2*args.d_model,
            device=device)

       
        head = TaskHead(
            input_dim= args.d_model+1,  
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None 
        )


@ModelBuilder.register("clf_tm_cv_attnpl_t")
class ClassifierTMConvSAttnPlTBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer_Conv
        from src.models.input_adapters import M_T_InputAdapterWithTime
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.extractors import RepresentationExtractor
        from src.models.base_model import BaseModel

        
        encoder = Transformer_Conv(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
        )

        input_adapter = M_T_InputAdapterWithTime()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)

        extractor = RepresentationExtractor.by_name("attn_time")(  
            input_dim= args.d_model+1,
            hidden_dim= 2*args.d_model,
            device=device)

       
        head = TaskHead(
            input_dim= args.d_model+1,  
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None 
        )


@ModelBuilder.register("classifier_stm_s")
class ClassifierTMSBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer
        from src.models.input_adapters import SM_T_InputAdapter
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.extractors import RepresentationExtractor
        from src.models.base_model import BaseModel

        
        encoder = Transformer(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            emb_n_layer= getattr(args,'emb_n_layers',4),
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
        )

        input_adapter = SM_T_InputAdapter()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)

        extractor = RepresentationExtractor.by_name("last")()

       
        head = TaskHead(
            input_dim= args.d_model,  
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None 
        )


@ModelBuilder.register("classifier_se")
class ClassifierSEBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer_SE
        from src.models.input_adapters import S_T_M_InputAdapter
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.extractors import RepresentationExtractor
        from src.models.base_model import BaseModel

        encoder = Transformer_SE(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            device=device,
            loc_dim=args.loc_dim,
            attn_type=args.attn_type,
        )

        # input adapter
        input_adapter = S_T_M_InputAdapter()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)
        extractor = RepresentationExtractor.by_name("last")()

        # classification head
        head = TaskHead(
            input_dim=3 * args.d_model,
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )
        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None
        )


@ModelBuilder.register("classifier_stm")
class ClassifierSTMBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer_STM
        from src.models.input_adapters import S_T_M_InputAdapter
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.extractors import RepresentationExtractor
        from src.models.base_model import BaseModel

        encoder = Transformer_STM(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            device=device,
            loc_dim=args.loc_dim,
            attn_type=args.attn_type,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
        )

        input_adapter = S_T_M_InputAdapter()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)

        extractor = RepresentationExtractor.by_name("last")()

        head = TaskHead(
            input_dim=4 * args.d_model,  # note
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None
        )


@ModelBuilder.register("clf_attnpl")
class ClassifierAttnPlBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer_ST
        from src.models.input_adapters import SM_T_InputAdapter
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.base_model import BaseModel
        from src.models.extractors.attention_pooling import AttentionPoolingExtractor

        # encoder
        encoder = Transformer_ST(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
        )

        input_adapter = SM_T_InputAdapter()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)

        # note
        extractor = AttentionPoolingExtractor(
            input_dim=3 * args.d_model,
            hidden_dim=3 * args.d_model,
            device=device
        )

        head = TaskHead(
            input_dim=3 * args.d_model,
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None
        )


@ModelBuilder.register("clf_attnpl_t")
class ClfAttnPlTBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer_ST
        from src.models.input_adapters import SM_T_InputAdapterWithTime
        from src.models.extractors.attn_pool_with_time import AttentionPoolingWithTimeExtractor
        from src.models.task_model import TaskModel
        from src.models.base_model import BaseModel
        from src.models.heads import TaskHead

        encoder = Transformer_ST(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            device=device,
            dim=args.dim,
            attn_type=args.attn_type,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0)
        )

        adapter = SM_T_InputAdapterWithTime()
        base_model = BaseModel(encoder=encoder, input_adapter=adapter, device=device)

        extractor = AttentionPoolingWithTimeExtractor(
            input_dim=3 * args.d_model + 1,
            hidden_dim=args.d_model,
            device=device
        )

        head = TaskHead(
            input_dim=3 * args.d_model + 1,
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None
        )


@ModelBuilder.register("lstm")
class LSTMBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.lstm import LSTM
        hidden_size = args.lstm_hidden_size
        num_layers = args.lstm_num_layers
        feature_size =  len(args.feature_cols)
        lstm_dropout = args.lstm_dropout
        output_activation = getattr(args, "output_activation", "sigmoid")
        if hasattr(args,'criterion_name') :
            output_size = len(args.criterion_cfg.taus) if args.criterion_cfg.get('taus', None) is not None else 1
        else:
            output_size = 1
        return LSTM(feature_size=feature_size, hidden_size=hidden_size, output_size=output_size,
                    num_layers=num_layers, lstm_dropout=lstm_dropout, device=device,
                    output_activation=output_activation)


@ModelBuilder.register("clf_mixer_attnpl_t")
class ClfMixerAttnPlTBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.input_adapters import MixerAdapter
        from src.models.extractors import RepresentationExtractor
        from src.models.extractors.attn_pool_with_time import AttentionPoolingWithTimeExtractor
        from src.models.extractors.attn_time_biased_mh import TimeAwareAttnPoolMH   
        from src.models.extractors.attn_time_biased import  TimeAwareAttnPool
        from src.models.extractors.pma_time_biased import TimeBiasedPMA
        from src.models.extractors.last_step import LastStepExtractor
        from src.models.task_model import TaskModel
        from src.models.mamba.mixer_seq import MixerModelWrapper, MixerModel
        from src.models.heads import TaskHead
        import torch

        encoder = MixerModel(**args.mixer_model_config, device=device, dtype=torch.float32).to(device)
        adapter = MixerAdapter(args)
        base_model = MixerModelWrapper(encoder=encoder, input_adapter=adapter, device=device)

        extractor_name = getattr(args, 'extractor_name') if hasattr(args, 'extractor_name') else 'attn_time'
        time_bias_type = getattr(args, 'time_bias_type', 'linear') if hasattr(args, 'time_bias_type') else 'linear'
        extractor_cfg = getattr(args, 'extractor_cfg', {})
        if extractor_name == 'attn_time':
            extractor = AttentionPoolingWithTimeExtractor(
                input_dim=args.d_model + 1,
                hidden_dim=args.d_model,
                device=device
            )
            head_input_dim = args.d_model + 1

        elif extractor_name == 'attn_time_biased':
            extractor = TimeAwareAttnPool(
                d_model=args.d_model,
                d_hidden=args.d_model,
                bias_type= time_bias_type,
                device=device,
                **extractor_cfg,
            )
            head_input_dim = args.d_model

        elif extractor_name == 'attn_time_biased_mh':
            n_heads = getattr(args, 'n_heads', 4)

            extractor = TimeAwareAttnPoolMH(
                d_model=args.d_model,
                d_hidden=args.d_model,
                bias_type=time_bias_type,
                n_heads=n_heads,
                agg=getattr(args, 'agg', 'concat'),
                device=device,
                **extractor_cfg
            )
            head_input_dim = extractor.output_dim

        elif extractor_name == 'pma_time_biased':
            n_heads = getattr(args, 'n_heads', 4)

            extractor = TimeBiasedPMA(
            d_model=args.d_model,
            n_heads= n_heads,
            r=getattr(args, 'pma_r', 4),
            agg=getattr(args, 'agg', 'mean'),
            use_film=getattr(args, 'use_film', True),
            bias_type=time_bias_type,
            alpha0=getattr(args, 'alpha0', 10.0),
            device=device
            )
            head_input_dim = args.d_model if getattr(args, 'agg', 'mean') == 'mean' else args.d_model * n_heads

        elif extractor_name == 'last':
            extractor = LastStepExtractor()
            head_input_dim = args.d_model
        else:
            raise ValueError(f"Unknown extractor_name: {extractor_name}")
        
        head = TaskHead(
            input_dim=head_input_dim,
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        revin_layer = None

        model = TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None,
            revin_layer=revin_layer
        )
        # Optional training-time magnitude noise for classification
        setattr(model, 'mag_noise_std', getattr(args, 'mag_noise_std', 0.0))
        setattr(model, 'mag_noise_type', getattr(args, 'mag_noise_type', 'gaussian'))
        return model


@ModelBuilder.register("clf_rnn")
class ClfRnnTBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.base_model import BaseModel
        from src.models.task_model import TaskModel
        from src.models.extractors.last_step import LastStepExtractor
        from src.models.heads import TaskHead
        from src.models.input_adapters import RTPPTaskInputAdapter
        from src.models.rnn_encoder import RNNEncoder, RNNEncoderWrapper

        input_adapter = RTPPTaskInputAdapter(args)
        encoder_core = RNNEncoder(
            input_dim=len(input_adapter.features_input_keys),
            hidden_dim=getattr(args, "d_model", 64),
            num_layers=getattr(args, "num_rnn_layers", 1),
            rnn_type=getattr(args, "rnn_type", "lstm"),
            dropout=getattr(args, "rnn_dropout", 0.0),
            bidirectional=bool(getattr(args, "bidirectional", False)),
        )
        encoder = RNNEncoderWrapper(encoder_core)
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)
        extractor = LastStepExtractor()
        head = TaskHead(
            input_dim=encoder.output_dim,
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device,
        )
        model = TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None,
        )
        setattr(model, 'mag_noise_std', getattr(args, 'mag_noise_std', 0.0))
        setattr(model, 'mag_noise_type', getattr(args, 'mag_noise_type', 'gaussian'))
        return model
