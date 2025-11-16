#import necessary libraries
import torch
import lightning as L
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from effdet import create_model


class EffDetLModel(L.LightningModule):

    def __init__(self, model_architecture, num_classes, bench_task="train", lr=0.0002, batch_size=8):
        super().__init__()

        self.save_hyperparameters("model_architecture", "num_classes", "bench_task", "lr", "batch_size")

        self.map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
        self.batch_size = batch_size
        self.lr = lr

        self.train_bool = (bench_task == "train")

        self.model_train = create_model(model_architecture, bench_task = "train", num_classes = num_classes, pretrained=True, bench_labeler=True)
        self.model_predict = create_model(model_architecture, bench_task = "predict", num_classes = num_classes, pretrained=False, bench_labeler=True)
        self.model_predict.load_state_dict(self.model_train.state_dict())
        

       
    def forward(self, images, targets=None):
        if (self.train_bool):
            print(f"self.model_train is called")
            return self.model_train(images, targets)
        else:
            print(f"self.model_predict called")
            return self.model_predict(images)
        

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model_train.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch

        losses = self.model_train(images, targets)


        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.batch_size)
        self.log(
            "train_class_loss", losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=self.batch_size
        )
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.batch_size)

        return losses['loss']
    
    def on_validation_start(self):
        self.model_predict.load_state_dict(self.model_train.state_dict())
        self.model_predict.eval()
        return super().on_validation_start()
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        #CGPT says that DetBenchPredict outputs xmin, ymin, xmax, ymax, score, label. 
        images, targets = batch

        # Ensure no grad
        preds_tensor = self.model_predict(images)

        # If preds_tensor is of shape [B, N, 6]:
        batch_size = preds_tensor.shape[0]

        pred_list = []
        for i in range(batch_size):
            single = preds_tensor[i]                     # shape [N,6]
            boxes  = single[:, :4]
            scores = single[:, 4]
            labels = single[:, 5].long()

            pred_list.append({
                "boxes":  boxes,
                "scores": scores,
                "labels": labels
            })

      

        target_list = []
        #iterate through batch size number of times. 
        for i in range(len(targets["bbox"])):

            #newly added
            boxes_xyxy = targets["bbox"][i].clone()
            boxes_xyxy[:, [0, 1, 2, 3]] = boxes_xyxy[:, [1, 0, 3, 2]]
            
            # convert target dict / tensor if necessary
            target_list.append({
                "boxes":  boxes_xyxy,
                "labels": targets["cls"][i]
            })


        # Now update metric
        self.map_metric.update(pred_list, target_list)

  
        
    def on_validation_epoch_end(self):

        map_metrics = self.map_metric.compute()
        self.log("mAP", map_metrics["map"], prog_bar=True, logger=True,batch_size=self.batch_size)
        self.log("mAP_50", map_metrics["map_50"], prog_bar=True, logger=True, batch_size=self.batch_size)
        print(map_metrics["map_50"])
        self.map_metric.reset()
        return super().on_validation_epoch_end()
    
   