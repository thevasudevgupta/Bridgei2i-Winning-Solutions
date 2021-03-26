import os
from transformers import MBartForConditionalGeneration, MBartTokenizer
from datasets import concatenate_datasets

from data_utils.dataloader import infer_bart_on_sample, DataLoader
from training_utils import Trainer, TrainerConfig
import training_utils

def summarize(sample, model, tokenizer, max_pred_length=32):
    sample["predicted_summary"] = infer_bart_on_sample(sample["CleanedText"], model, tokenizer, max_pred_length)
    return sample

# def assign_split(x, samples):
#     x["split"] = None
#     if x['Text_ID'] in samples:
#         x["split"] = "TRAIN"
#     else:
#         x["split"] = "VALIDATION"
#     return x

if __name__ == '__main__':

    args = TrainerConfig.from_default()
    args.update(getattr(training_utils, "baseline").__dict__)

    model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model_id)
    tokenizer = MBartTokenizer.from_pretrained(args.pretrained_tokenizer_id)

    dl = DataLoader(tokenizer, args)
    tr_dataset, val_dataset = dl.setup(process_on_fly=args.process_on_fly, n_augment=args.n_augment)
    print(tr_dataset, val_dataset)

    tr_dataset = dl.train_dataloader(tr_dataset)
    val_dataset = dl.val_dataloader(val_dataset)

    # tr_dataset = [next(iter(tr_dataset)) for i in range(6)]
    # val_dataset = [next(iter(val_dataset)) for i in range(6)]

    trainer = Trainer(model, args)
    trainer.fit(tr_dataset, val_dataset)

    fn_kwargs = {
      "model": model,
      "tokenizer": tokenizer,
      "max_pred_length": 44,
    }
    combined_data = concatenate_datasets([tr_dataset, val_dataset]).sort('Text_ID')
    combined_data = combined_data.filter(lambda x: x["augmentation_status"] == "Not Augmented")
    print("COMBINED DATA", combined_data)
    combined_data = combined_data.map(summarize, fn_kwargs=fn_kwargs)

    # samples = []
    # for s in tr_dataset:
    #     samples.append(s['Text_ID'])

    # combined_data = combined_data.map(assign_split, fn_kwargs=dict(samples=samples))
    print(combined_data)
    combined_data.to_csv(os.path.join(args.base_dir, "predictions.csv"))
