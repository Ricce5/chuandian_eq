from src.utils.registrable import Registrable

class ModelBuilder(Registrable):
    def __call__(self, args, device):
        raise NotImplementedError


# models/builders.py

@ModelBuilder.register("classifier")
class ClassifierBuilder(ModelBuilder):
    def __call__(self, args, device):
        from models.transformer import Transformer_ST
        from models.input_adapters import SM_T_InputAdapter
        from models.task_model import TaskModel
        from models.heads import TaskHead
        from models.extractors import RepresentationExtractor
        from models.base_model import BaseModel

        encoder = Transformer_ST(...)

        base_model = BaseModel(encoder, SM_T_InputAdapter(), device)

        extractor = RepresentationExtractor.by_name("last")()

        head = TaskHead(
            input_dim=3 * args.d_model,
            output_dim=args.mlp_out,
            head_type="mlp",
            hidden_layers=args.mlp_hdw,
            dropout=args.mlp_dropout
        )

        return TaskModel(base_model, extractor, head, final_activation=None)
