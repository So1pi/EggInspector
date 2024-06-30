# Egg Inspector

This repository implements egg object detection using Faster R-CNN with a ResNet-50 backbone.

![image](https://github.com/So1pi/EggInspector/assets/173986541/c27eed6d-7f54-4f0f-903a-5e20aa33bed5)
![image](https://github.com/So1pi/EggInspector/assets/173986541/5915c3fd-f040-4540-86b7-6d2975e1e896)



## Configuration

- **Number of GPUs:** 4
- **Architecture:**
  - Type: Faster R-CNN
  - Backbone: ResNet-50
- **Number of Classes:** 6

## Data Loader

- **Type:** EggDataLoader
- **Arguments:**
  - Data Directory: "data/"
  - Batch Size: 1024
  - Shuffle: True
  - Validation Split: 0.1
  - Number of Workers: 4

## Optimizer

- **Type:** SGD
- **Arguments:**
  - Learning Rate: 0.005
  - Momentum: 0.9
  - Weight Decay: 0.005

## Loss Function

- Cross-Entropy Loss

## Evaluation Metrics

- Precision, Recall, F1 Score

## Learning Rate Scheduler

- **Type:** StepLR
- **Arguments:**
  - Step Size: 50
  - Gamma: 0.1

## Data Transformations

### Training
![image](https://github.com/So1pi/EggInspector/assets/173986541/71825007-2ee1-4e46-95f6-9703b4865e55)

- Random Rotation: [-45, 45] degrees
- Color Adjustments: Brightness=0.5, Contrast=0.5, Saturation=0.5, Hue=0.5

### Testing

- Resize: Size=256
- Center Crop: Size=224

## Trainer

- **Epochs:** 10
- **Save Directory:** "saved/"
- **Save Frequency:** Save model every epoch
- **Verbosity:** Level 2
- **Monitoring:** Minimize Validation Loss
- **Early Stopping:** Wait 10 epochs before stopping
- **TensorBoard:** Enabled

## Miscellaneous

- **Data Path:** [AI-hub Egg Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71504)
- **Batch Size:** 4
- **Collate Function:** collate_fn_custom
- **Number of Classes:** 6
- **Class Weights:** [1.0, 1.5, 1.0, 1.0, 1.0, 1.0]

---

This project aims to effectively detect various egg objects using Faster R-CNN with a ResNet-50 backbone. The provided configuration includes essential elements for training and evaluating the model on a specific dataset.
