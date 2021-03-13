
from dataclasses import dataclass, field, replace

@dataclass
class TrainingArgs:

    lr: float = 1.e-5
    batch_size: int = 2
    max_epochs: int = 10
    accumulation_steps: int = 4

    num_workers: int = 2
    max_length: int = 512
    max_target_length: int = 32
    process_on_fly: bool = False

    file_path: str = "data/dev_data_article.csv"
    pretrained_model_id: str = "facebook/mbart-large-cc25"
    weights_dir: str = "mbart-finetuned-summary"
    # hub_id: str = "mbart"

    base_dir: str = "mbart-summarizer"
    wandb_run_name: str = "dummy"
    project_name: str = "interiit-mbart"

baseline = TrainingArgs()
