import torchvision
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from transformers import DetrImageProcessor
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
from pytorch_lightning import Trainer

print(torch.cuda.current_device())
print(torch.cuda.is_available())

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, data_folder, processor, train=True):
        ann_file = os.path.join(data_folder, "../annotations", "train.json" if train else "test.json")
        super(CocoDetection, self).__init__(data_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
    
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader
     

if __name__ == "__main__": 
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", max_size=640)

    train_dataset = CocoDetection(data_folder='/storage/anik/dataset/doclaynet_COCO/train', processor=processor)
    val_dataset = CocoDetection(data_folder='/storage/anik/dataset/doclaynet_COCO/test', processor=processor, train=False)

    # image_ids = train_dataset.coco.getImgIds()
    # # let's pick a random image
    # image_id = image_ids[np.random.randint(0, len(image_ids))]
    # image = train_dataset.coco.loadImgs(image_id)[0]
    # image = Image.open(os.path.join('/storage/anik/dataset/doclaynet_COCO/train', image['file_name']))

    # annotations = train_dataset.coco.imgToAnns[image_id]
    # draw = ImageDraw.Draw(image, "RGBA")

    cats = train_dataset.coco.cats

    id2label = {k: v['name'] for k,v in cats.items()}

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=50, shuffle=True, num_workers=11)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=50, num_workers=11)
    batch = next(iter(train_dataloader))

    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

    torch.set_float32_matmul_precision("high")
    # trainer = Trainer(max_steps=17340, gradient_clip_val=0.1, devices=2)
    # trainer = Trainer(default_root_dir= "checkpoints", enable_checkpointing=True, max_steps=6940, gradient_clip_val=0.1, devices=2)
    trainer = Trainer(default_root_dir= "checkpoints", enable_checkpointing=True, max_steps=20820, gradient_clip_val=0.1, devices=2)
    # trainer = Trainer(max_steps=300, gradient_clip_val=0.1, accelerator="gpu", devices=2)
    trainer.fit(model)
    trainer.save_checkpoint("detr_all_new.pth")