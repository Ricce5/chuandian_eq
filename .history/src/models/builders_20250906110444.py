from src.utils.registrable import Registrable

class ModelBuilder(Registrable):
    def __call__(self, args, device):
        raise NotImplementedError


@ModelBuilder.register("classifier")
class ClassifierBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer_ST
        from src.models.input_adapters import SM_T_InputAdapter
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.extractors import RepresentationExtractor
        from src.models.base_model import BaseModel

        # 构建 Transformer 编码器
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

        # 输出 head（多层感知机）
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

        # 构建 Transformer 编码器
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

        # 输出 head（多层感知机）
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

        # 构建 Transformer 编码器
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

        # 输出 head（多层感知机）
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

        # 构建 Transformer 编码器
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

        # 输出 head（多层感知机）
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

        # 构建 Transformer 编码器
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

        # 输出 head（多层感知机）
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

        # 构建 Transformer 编码器
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

        # 输出 head（多层感知机）
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

        # 输入适配器
        input_adapter = S_T_M_InputAdapter()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)
        extractor = RepresentationExtractor.by_name("last")()

        # 分类头（输入维度为 3 * d_rnn）
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
            input_dim=4 * args.d_model,  # 注意这里是 4，因为 STM 有多分支拼接输出
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

        # 编码器
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

        # 注意这里 input_dim 不带时间：transformer 输出是 [B, L, 3*d_model]
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
        output_size = getattr(args, 'output_size', 1) 
        num_layers = args.lstm_num_layers
        feature_size =  len(args.feature_cols)
        return LSTM(args, device=device)



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
        return RecurrentTPP(args, device)


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
    


@ModelBuilder.register("reg_attnpl")
class RegressorAttnPlBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.transformer import Transformer_ST
        from src.models.input_adapters import SM_T_InputAdapter
        from src.models.task_model import TaskModel
        from src.models.heads import TaskHead
        from src.models.base_model import BaseModel
        from src.models.extractors.attention_pooling import AttentionPoolingExtractor
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

        # 注意这里 input_dim 不带时间：transformer 输出是 [B, L, 3*d_model]
        extractor = AttentionPoolingExtractor(
            input_dim=3 * args.d_model,
            hidden_dim=3 * args.d_model,
            device=device
        )

        head = TaskHead(
            input_dim=3 * args.d_model,
            output_dim=1,  # For regression, output is typically a single continuous value
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout,
            device=device
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None  # No activation for regression output (just raw value)
        )

@ModelBuilder.register("mixer_tpp")
class MixerTPPBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.tpp.mixer_tpp import MixerTPP
        from src.models.input_adapters import Mixer_BatchInputAdapter
        from src.models.heads import TaskHead
        from src.models.base_model import BaseModel
        from src.models.mamba.mixer_seq import MixerModel,MixerModelWrapper
        from src.models.mamba.scan_wrapper import BoundedSelectiveScanWrapper,BoundedDiscreteSSM
        import torch
        import torch.nn as nn   
        encoder = MixerModel(**args.mixer_model_config, device=device, dtype=torch.float32).to(device)
        adapter = Mixer_BatchInputAdapter(args)
        hypernet_time = nn.Linear(args.d_model, 3 * args.num_components).to(device)
        hypernet_mag = nn.Linear(args.d_model, 1).to(device)
        if getattr(args, 'use_ssm_filter', True):
            # ssm_filter = BoundedSelectiveScanWrapper(d_model=1, d_state=1, device=device, 
            # B_range=getattr(args,'B_range',(1e-6, 1e-3)), output_range=getattr(args,'b_range',(0.5,2))).to(device) 
            ssm_filter = BoundedDiscreteSSM(device=device,B_range=getattr(args,'B_range',(1e-6, 1e-3)),
                         output_range=getattr(args,'b_range',(0.5,2)),output_init=args.richter_b_mle).to(device) 
        else:
            ssm_filter = None
        base_model = MixerModelWrapper(encoder=encoder, input_adapter=adapter, device=device)
        predict_b = getattr(args, 'predict_b', False)
        use_b_updater = getattr(args, 'use_b_updater', False)
        loss_weights = getattr(args, 'loss_weights', None)
        loss_reduction = getattr(args, 'loss_reduction', None)
        use_adaptive_loss_weights = getattr(args, 'use_adaptive_loss_weights', False)
        b_range = getattr(args, 'b_range', None)
        return MixerTPP(base_model, hypernet_time, hypernet_mag, dropout=args.dropout,
                        predict_b=predict_b, ssm_filter=ssm_filter,use_b_updater=use_b_updater,
                        loss_weights=loss_weights, loss_reduction=loss_reduction,b_range=b_range,
                        use_adaptive_loss_weights=use_adaptive_loss_weights)



@ModelBuilder.register("clf_mixer_attnpl_t")
class ClfMixerAttnPlTBuilder(ModelBuilder):
    def __call__(self, args, device):
        from src.models.input_adapters import MixerInputAdapterWithTime
        from src.models.extractors import RepresentationExtractor
        from src.models.extractors.attn_pool_with_time import AttentionPoolingWithTimeExtractor
        from src.models.extractors.attn_time_biased_mh import TimeAwareAttnPoolMH   
        from src.models.extractors.attn_time_biased import  TimeAwareAttnPool
        from src.models.extractors.pma_time_biased import TimeBiasedPMA
        from src.models.task_model import TaskModel
        from src.models.mamba.mixer_seq import MixerModelWrapper, MixerModel
        from src.models.heads import TaskHead
        import torch

        encoder = MixerModel(**args.mixer_model_config, device=device, dtype=torch.float32).to(device)
        adapter = MixerInputAdapterWithTime(args)
        base_model = MixerModelWrapper(encoder=encoder, input_adapter=adapter, device=device)

        extractor_name = getattr(args, 'extractor_name') if hasattr(args, 'extractor_name') else 'attn_time'
        time_bias_type = getattr(args, 'time_bias_type', 'linear') if hasattr(args, 'time_bias_type') else 'linear'
        extractor_cfg = getattr(args, 'extractor_cfg', {})
        print('using extractor:', extractor_name, "time_bias_type:", time_bias_type, "extractor_config:", extractor_cfg)
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

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None
        )

@ModelBuilder.register("reg_mixer_attnpl_t")
class RegMixerAttnPlTBuilder(ModelBuilder):
    def __call__(self, args, device):
            from src.models.input_adapters import MixerInputAdapterWithTime
            from src.models.extractors import RepresentationExtractor
            from src.models.extractors.attn_pool_with_time import AttentionPoolingWithTimeExtractor
            from src.models.extractors.attn_time_biased_mh import TimeAwareAttnPoolMH   
            from src.models.extractors.attn_time_biased import  TimeAwareAttnPool
            from src.models.extractors.pma_time_biased import TimeBiasedPMA
            from src.models.task_model import TaskModel
            from src.models.mamba.mixer_seq import MixerModelWrapper, MixerModel
            from src.models.heads import TaskHead
            import torch

            encoder = MixerModel(**args.mixer_model_config, device=device, dtype=torch.float32).to(device)
            adapter = MixerInputAdapterWithTime(args)
            base_model = MixerModelWrapper(encoder=encoder, input_adapter=adapter, device=device)

            extractor_name = getattr(args, 'extractor_name') if hasattr(args, 'extractor_name') else 'attn_time'
            time_bias_type = getattr(args, 'time_bias_type', 'linear') if hasattr(args, 'time_bias_type') else 'linear'
            extractor_cfg = getattr(args, 'extractor_cfg', {})
            print('using extractor:', extractor_name, "time_bias_type:", time_bias_type, "extractor_config:", extractor_cfg)
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

            return TaskModel(
                base_model=base_model,
                extractor=extractor,
                head=head,
                final_activation=None
            )