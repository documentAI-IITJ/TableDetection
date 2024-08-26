import torchvision, torch
import pytorch_lightning as pl
from transformers import DetrImageProcessor, DetrForObjectDetection
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw
import json

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
     
# Define the function to load the checkpoint and model
def load_model(checkpoint_path):
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# Function to save predictions to CSV
def save_predictions_to_csv(predictions, filename="predictions_on_test.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "bbox", "score", "label"])
        for prediction in predictions:
            writer.writerow(prediction)
     

# Define color dictionary for each label
label_colors = {
    0: (255, 0, 0),    # Red
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Blue
    3: (255, 255, 0),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Cyan
    6: (128, 0, 0),    # Maroon
    7: (0, 128, 0),    # Olive
    8: (0, 0, 128),    # Navy
    9: (128, 128, 0),  # Olive Green
    10: (0, 128, 128)  # Teal
}

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def plot_results(img_path, scores, labels, boxes, output_dir, image_filename, id2label):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    for score, label, (xmin, ymin, xmax, ymax)  in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
    # Get color for label
        if int(label) in label_colors:
            color = label_colors[int(label)]
        else:
            color = get_random_color()  # Assign random color for labels not in label_colors
        
        # Draw bounding box with label color
        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Add label text
        label_text = str(id2label[int(label)])
        cv2.putText(image, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Save the image with annotations to the output directory
    output_path = os.path.join(output_dir, image_filename)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def convert_to_json(filename, cls_list, conf_list, xyxy_list):
    label_map = create_label_mapping()
    json_data = {
        "filename" : filename,
        "detections": [
            {
                "cls": int(cls_list[i]),
                "cls_name" : label_map.get(int(cls_list[i]), "Unknown label"),
                "conf": float(conf_list[i]),
                "bbox": 
                {
                    "x1": float(xyxy_list[i][0]),
                    "y1": float(xyxy_list[i][1]),
                    "x2": float(xyxy_list[i][2]),
                    "y2": float(xyxy_list[i][3])
                },
            }
            for i in range(len(cls_list))
        ]
    }
    return json.dumps(json_data, indent=4)



if __name__ == "__main__": 
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", max_size=640)

    # train_dataset = CocoDetection(data_folder='/storage/anik/dataset/doclaynet_COCO/train', processor=processor)
    val_dataset = CocoDetection(data_folder='/storage/anik/dataset/doclaynet_COCO/test', processor=processor, train=False)
    cats = val_dataset.coco.cats

    id2label = {k: v['name'] for k,v in cats.items()}
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=10, num_workers=11)

    # Load the model
    checkpoint_path = "prev_checkpoints/lightning_logs/version_2/checkpoints/epoch=29-step=20820.ckpt"
    model = load_model(checkpoint_path)

    # Collect all predictions
    all_predictions = []
    count = 0
    # Predict on the test set
    model.eval()  # Set the model to evaluation mode
    for pixel_values, target in tqdm(val_dataset):
        pixel_values = pixel_values.unsqueeze(0)
        outputs = model(pixel_values=pixel_values, pixel_mask=None)

        image_id = target['image_id'].item()
        image_info = val_dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join('/storage/anik/dataset/doclaynet_COCO/test', image_info['file_name'])
        image = Image.open(os.path.join('/storage/anik/dataset/doclaynet_COCO/test', image_info['file_name']))

        # Post-process model outputs
        width, height = image.size
        postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                        target_sizes=[(height, width)],
                                                                        threshold=0.9)
        results = postprocessed_outputs[0]
        bboxes = results['boxes'].tolist()
        scores = results['scores'].tolist()
        labels = results['labels'].tolist()
        # plot_results(image_path, results['scores'], results['labels'], results['boxes'], "/storage/anik/docai/notebooks/doclaynet_test", image_info['file_name'], id2label)
        # Combine bbox, score, and label into the required format
        # formatted_predictions = [
        #     [bbox[0], bbox[1], bbox[2], bbox[3], score, label]
        #     for bbox, score, label in zip(bboxes, scores, labels)
        # ]
        # json_string = convert_to_json(image_info['file_name'], )
        print(postprocessed_outputs)
        print(results)
        break

    #     # Add the image_id to each prediction
    #     for pred in formatted_predictions:
    #         all_predictions.append([image_info['file_name']] + pred)
    #     if count > 100:
    #         break
    #     count += 1
    # # Save all predictions to CSV
    # save_predictions_to_csv(all_predictions)