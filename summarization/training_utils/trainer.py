
import os
import torch
from .torch_trainer import TorchTrainer


class Trainer(TorchTrainer):

    def __init__(self, model, args):

        self.model = model
        self.args = args
        super().__init__(args)

    def fetch_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def training_step(self, batch, batch_idx):

        for k in batch:
            batch[k] = batch[k].to(self.device)

        out = self.model(**batch, return_dict=True)
        loss = out["loss"].mean()
        return loss

    @torch.no_grad()
    def validation_step(self, batch):

        for k in batch:
            batch[k] = batch[k].to(self.device)

        out = self.model(**batch, return_dict=True)

        loss = out["loss"].mean()
        return loss

    def training_epoch_end(self, epoch, losses):
        # saving state_dict at epoch level
        if self.args.weights_dir:
            self.model.save_pretrained(os.path.join(self.args.base_dir, self.args.weights_dir+f"-e{epoch}"))
