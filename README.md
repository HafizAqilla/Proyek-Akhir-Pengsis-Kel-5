Of course. Here is the complete README content in proper `.md` (Markdown) format. You can copy and paste this directly into a `README.md` file in your GitHub repository.

---

# DeepChest AI: Vision Transformer & YOLOv8 for Tuberculosis Detection

This project harnesses the power of advanced deep learning architectures—the Vision Transformer (ViT) for high-accuracy classification and YOLOv8 for precise object detection—to analyze chest X-ray images for signs of Tuberculosis (TB). By creating a dual-pronged diagnostic pipeline, we aim to provide a robust, automated tool that can classify images and localize potential infections, serving as a powerful aid in medical imaging analysis.

## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and testing purposes.

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
    git clone https://github.com/your-username/DeepChest-AI.git
    cd DeepChest-AI
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
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download the dataset.** This project uses the TBX11K dataset. Download it from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) and place it in a `data/` directory, ensuring the `tbx11k-simplified` folder is present.

## Executing the Pipeline

The project is split into two main pipelines: Vision Transformer for classification and YOLOv8 for detection.

### 1. Vision Transformer Pipeline

This script prepares the data and trains the ViT classification model.

First, update the `base_path` variable in `vit_training_pyasd.py` to point to your `tbx11k-simplified` directory.

Then, execute the script:
```sh
python vit_training_pyasd.py
```
This will run the full pipeline: data validation, dataset splitting, training, and evaluation. Final metrics, plots, and the trained model (`vit_model.pth`) will be saved.

### 2. YOLOv8 Pipeline

This pipeline is a two-step process.

**Step 1: Prepare the Data for YOLO**

Run the data preparation script to convert the dataset into the required YOLO format. Update the paths in the script first.
```sh
# Assuming Untitled-1.py is renamed to yolo_prepare_data.py
python yolo_prepare_data.py  
```
This creates the `yolo_training_tb_only` directory with `images`, `labels`, and a `data.yaml` file.

**Step 2: Train the YOLOv8 Model**

Execute the training script.
```sh
# Assuming the training logic is separated into yolo_train.py
python yolo_train.py 
```
Upon completion, the training run will be saved in a directory like `yolo_training_tb_only/yolov8m_tb_training/run1`, containing performance plots, logs, and the best model weights (`best.pt`).

## Built With

- [PyTorch](https://pytorch.org/) - The core deep learning framework.
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - For the object detection model and pipeline.
- [Scikit-learn](https://scikit-learn.org/) - For data splitting and evaluation metrics.
- [Pandas](https://pandas.pydata.org/) - For data manipulation.

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your-username/DeepChest-AI/tags).

## Authors

- **Your Name** - *Initial Work* - [YourGitHubProfile](https://github.com/your-username)

See also the list of [contributors](https://github.com/your-username/DeepChest-AI/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Hat tip to the creators of the [TBX11K Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset).
- Inspiration from the broader medical AI research community.
