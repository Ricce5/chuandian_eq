from src.utils.registrable import Registrable

class ModelBuilder(Registrable):
    def __call__(self, args, device):
        raise NotImplementedError


@ModelBuilder.register("classifier")
class ClassifierBuilder(ModelBuilder):
    def __call__(self, args, device):
        from models.transformer import Transformer_ST
        from models.input_adapters import SM_T_InputAdapter
        from models.task_model import TaskModel
        from models.heads import TaskHead
        from models.extractors import RepresentationExtractor
        from models.base_model import BaseModel

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
            loc_dim=args.dim,
            attn_type=args.attn_type,
        )

        input_adapter = SM_T_InputAdapter()
        base_model = BaseModel(encoder=encoder, input_adapter=input_adapter, device=device)

        extractor = RepresentationExtractor.by_name("last")()

        # 输出 head（多层感知机）
        head = TaskHead(
            input_dim=3 * args.d_model,  # 输入维度 = Transformer 输出 * 3（默认结构）
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout
        )

        return TaskModel(
            base_model=base_model,
            extractor=extractor,
            head=head,
            final_activation=None  # 使用 BCEWithLogitsLoss 不需要 sigmoid
        )
