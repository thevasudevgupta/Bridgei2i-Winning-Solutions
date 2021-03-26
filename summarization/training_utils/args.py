
from dataclasses import dataclass, field, replace

@dataclass
class TrainingArgs:

    lr: float = 1.e-5
    batch_size: int = 12
    max_epochs: int = 10
    accumulation_steps: int = 1

    num_workers: int = 2
    max_length: int = 512
    max_target_length: int = 32
    process_on_fly: bool = True
    seed: int = 42
    n_augment: int = 2

    file_path: str = "/home/sneezygiraffe/interiit/clean_article.csv"
    pretrained_model_id: str = "facebook/mbart-large-cc25"
    pretrained_tokenizer_id: str = "facebook/mbart-large-cc25"
    weights_dir: str = "mbart-augmented-summary"
    # hub_id: str = "mbart"

    base_dir: str = "mbart-augmented-summary"
    wandb_run_name: str = "mbart-augmented-summary"
    project_name: str = "interiit-mbart"

baseline = TrainingArgs()
