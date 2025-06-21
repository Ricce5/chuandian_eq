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
            input_dim=3 * args.d_rnn,
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

@ModelBuilder.register("classifier_attnpl")
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
            attn_type=args.attn_type
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
        return LSTM(args, device=device)
