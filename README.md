
# Deteksi dan Klasifikasi Tuberkulosis (TBC) pada Citra Rontgen Dada menggunakan Vision Transformer (ViT) dan You Only Look Once v8 (Yolo v8)

## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and testing purposes. As for its final version, it is deployed on Hugging face so it can be run without any afromentioned prequisites.

### Prerequisites

Requirements for the software and other tools to build, test, and run the models:
- [Python 3.10+](https://www.python.org/)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- An NVIDIA GPU with CUDA support is highly recommended for training.

### Installing

A step-by-step guide to get your development environment running.

1.  **Clone the repository** to your local machine.
    ```sh
    git clone https://github.com/HafizAqilla/Proyek-Akhir-Pengsis-Kel-5
    ```

2.  **Create and activate a virtual environment** (recommended).
    ```sh
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages.**

4.  **Download the dataset.** This project uses the TBX11K dataset. Download it from [Kaggle] (https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified/data) and place it in a directory ensuring the `tbx11k-simplified` folder is present.

## Executing the Pipeline

The project is split into two main pipelines: Vision Transformer for classification and YOLOv8 for detection.

### 1. Vision Transformer Pipeline

This script prepares the data and trains the ViT classification model.

First, update the `base_path` variable in `YOLOv8Training.ipynb` to point to your `tbx11k-simplified` directory.

Then execute the script VitTraining on Jupiter Notebook

This will run the full pipeline: data validation, dataset splitting, training, and evaluation. Final metrics, plots, and the trained model (`vit_model.pth`) will be saved.

### 2. YOLOv8 Pipeline

This pipeline is a two-step process.

**Step 1: Prepare the Data for YOLO**

Run the data preparation script to convert the dataset into the required YOLO format. Update the paths in the script first.
```sh
# === 2. Cleanup and Directory Creation ===
if os.path.exists(base_dir):
    print(f"🧹 Cleaning up old directory: {base_dir}")
    rmtree(base_dir)

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
print(f"📁 Created YOLO directory structure in: {base_dir}")
```
This creates the `yolo_training_tb_only` directory with `images`, `labels`, and a `data.yaml` file.

**Step 2: Train the YOLOv8 Model**

Execute the training script.

Upon completion, the training run will be saved in a directory like `yolo_training_tb_only/yolov8m_tb_training/run1`, containing performance plots, logs, and the best model weights (`best.pt`).

## Running the GUI

To make it accessible to broader audiences, we deployed the final version on Hugging Face (https://huggingface.co/spaces/sherlyangel/UAS_Pengsis_TBC_Detector). 

Simply upload the X-ray images and it will tell you the probability and if it detects it as TBC, it will also create bounding boxes that detect the TBC lesion on lungs.

For running the final GUI on your local machine, make sure the .pt weight files are in the same directory as your GUI to make sure it running smoothly

## Notes

To see our Demo Video and Perfomance database, see [drive](https://drive.google.com/drive/folders/1uF9R3L1M5hSjoT9rKVr37sWe1K8KPfCy?usp=sharing) 

## Built With

- [PyTorch](https://pytorch.org/) - The core deep learning framework.
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - For the object detection model and pipeline.
- [Scikit-learn](https://scikit-learn.org/) - For data splitting and evaluation metrics.
- [Pandas](https://pandas.pydata.org/) - For data manipulation.

## Authors

- **Muhammad Hafiz Aqilla Subagyo** 
- **Sherly Angel Zuliany** - 
- **Albertus Satrio Aditama Christiyanto** 


## Acknowledgments

- Hat tip to the creators of the [TBX11K Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset).
- Our special thanks to Mohammad Ikhsan, PhD for his invaluable support and guidance throughout the development of this final project.
