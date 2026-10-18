# Task 3: Assembly Line Defect Detection (Computer Vision)

**Live demo:** https://growfinix-task3-defect-vision.streamlit.app/

Part of the Growfinix Technology Machine Learning Internship (Month 2).

Classifies photos of manufactured parts as **Defective** or **OK** using transfer learning: an ImageNet-pretrained ResNet-18 fine-tuned in PyTorch.

## Dataset
The `casting_512x512` version of Kaggle's "Real-life industrial dataset of casting product" (top-view photos of submersible pump impeller castings, about 1,300 images with `def_front` and `ok_front` classes). The dataset is not stored in this repo. Download it from Kaggle and unzip it into `data/`. Six sample images are included in `samples/` for the demo app. Please check the dataset's license on its Kaggle page.

## Method
- Stratified 80/20 split (seed 42): 1,040 training and 260 test images.
- Training augmentation: random horizontal flip and small rotation. Images are resized to 224x224 and normalized with ImageNet statistics.
- Phase 1: freeze the pretrained ResNet-18 backbone and train only a new 2-class head (3 epochs).
- Phase 2: unfreeze the last block (`layer4`) and fine-tune it with a lower learning rate (2 epochs).
- The app uses OpenCV (Gaussian blur and Canny edge detection) to show an edge view of the part next to the prediction. This view is for inspection only. The classifier itself uses the torchvision pipeline.
- The test set is used once, for the final evaluation only.

## Results
Held-out test set (260 images):

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Defective | 1.0000 | 0.9872 | 0.9935 | 156 |
| OK | 0.9811 | 1.0000 | 0.9905 | 104 |

Overall accuracy: 0.9923

Confusion matrix: 154 of the 156 defective parts were caught and 2 were missed. All 104 OK parts were classified correctly. In a factory, a missed defect is the costly error, so defect recall is the key number.

## Limitations
- The test set is small (260 images), so the scores could shift by a point or two with a different split.
- All photos show the same type of part under very consistent lighting and angle, so accuracy on real production lines would likely be lower.
- This is a learning project and not a certified inspection system.

## Run it
Use Python 3.11.

    pip install -r requirements.txt
    python train_defect.py
    streamlit run app.py

## Files
- `train_defect.py`: training and evaluation script
- `app.py`: Streamlit app (sample images or your own upload)
- `models/defect_resnet18.pt`: trained weights
- `outputs/`: test metrics and confusion matrix
- `samples/`: demo images
