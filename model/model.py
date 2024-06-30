import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def get_model_instance_segmentation(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
