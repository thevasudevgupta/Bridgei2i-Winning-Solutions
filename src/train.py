
from transformers import MBartForConditionalGeneration, MBartTokenizer
from data_utils import DataLoader

from training_utils import Trainer, TrainerConfig
import training_utils

if __name__ == '__main__':

    args = TrainerConfig.from_default()
    args.update(getattr(training_utils, "baseline").__dict__)

    model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model_id)
    tokenizer = MBartTokenizer.from_pretrained(args.pretrained_model_id)

    dl = DataLoader(tokenizer, args)
    tr_dataset, val_dataset = dl.setup()
    tr_dataset = dl.train_dataloader(tr_dataset)
    val_dataset = dl.val_dataloader(val_dataset)

    trainer = Trainer(model, args)
    trainer.fit(tr_dataset, val_dataset)
