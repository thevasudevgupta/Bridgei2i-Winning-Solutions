from dataclasses import dataclass, field, replace

@dataclass
class TrainingArgs:

    lr: float = 1.e-5
    batch_size: int = 2
    num_workers: int = 2

    max_length: int = 512
    max_target_length: int = 20

    file_path: str = "data/dev_data_article.csv"
    pretrained_model_id: str = "facebook/mbart-large-cc25"
    weights_dir: str = "mbart-finetuned"
    hub_id: str = "dummy"

    base_dir: str = None

baseline = TrainingArgs()
