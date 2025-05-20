import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

class BaseModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
    def get_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def get_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        raise NotImplementedError

class ClassificationModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_classes = config['num_classes']
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def get_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.CrossEntropyLoss()(pred, target)
        
    def get_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pred_labels = torch.argmax(pred, dim=1)
        accuracy = (pred_labels == target).float().mean().item()
        return {'accuracy': accuracy}

class SegmentationModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_classes = config['num_classes']
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def get_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.BCEWithLogitsLoss()(pred, target)
        
    def get_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pred_probs = torch.sigmoid(pred)
        pred_masks = (pred_probs > 0.5).float()
        
        intersection = (pred_masks * target).sum().item()
        union = pred_masks.sum().item() + target.sum().item() - intersection
        
        iou = intersection / (union + 1e-8)
        dice = (2 * intersection) / (pred_masks.sum().item() + target.sum().item() + 1e-8)
        
        return {
            'iou': iou,
            'dice': dice
        } 