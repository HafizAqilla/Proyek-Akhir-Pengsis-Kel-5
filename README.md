DeepChest AI: Vision Transformer & YOLOv8 for Tuberculosis Detection
This project harnesses the power of advanced deep learning architectures—the Vision Transformer (ViT) for high-accuracy classification and YOLOv8 for precise object detection—to analyze chest X-ray images for signs of Tuberculosis (TB). By creating a dual-pronged diagnostic pipeline, we aim to provide a robust, automated tool that can classify images and localize potential infections, serving as a powerful aid in medical imaging analysis.
Getting Started
These instructions will give you a copy of the project up and running on your local machine for development and testing purposes.
Prerequisites
Requirements for the software and other tools to build, test, and run the models:
Python 3.10+
PyTorch
Ultralytics
An NVIDIA GPU with CUDA support is highly recommended for training.
Installing
A step-by-step guide to get your development environment running.
Clone the repository to your local machine.
git clone https://github.com/your-username/DeepChest-AI.git
cd DeepChest-AI
Use code with caution.
Sh
Create and activate a virtual environment (recommended).
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Use code with caution.
Sh
Install the required packages.
pip install -r requirements.txt
Use code with caution.
Sh
Download the dataset. This project uses the TBX11K dataset. Download it from Kaggle and place it in a data/ directory, ensuring the tbx11k-simplified folder is present.
Executing the Pipeline
The project is split into two main pipelines: Vision Transformer for classification and YOLOv8 for detection.
1. Vision Transformer Pipeline
This script prepares the data and trains the ViT classification model.
First, update the base_path variable in vit_training_pyasd.py to point to your tbx11k-simplified directory.
Then, execute the script:
python vit_training_pyasd.py
Use code with caution.
Sh
This will run the full pipeline: data validation, dataset splitting, training, and evaluation. Final metrics, plots, and the trained model (vit_model.pth) will be saved.
2. YOLOv8 Pipeline
This pipeline is a two-step process.
Step 1: Prepare the Data for YOLO
Run the data preparation script to convert the dataset into the required YOLO format. Update the paths in the script first.
# Assuming Untitled-1.py is renamed to yolo_prepare_data.py
python yolo_prepare_data.py
Use code with caution.
Sh
This creates the yolo_training_tb_only directory with images, labels, and a data.yaml file.
Step 2: Train the YOLOv8 Model
Execute the training script.
# Assuming the training logic is separated into yolo_train.py
python yolo_train.py
Use code with caution.
Sh
Upon completion, the training run will be saved in a directory like yolo_training_tb_only/yolov8m_tb_training/run1, containing performance plots, logs, and the best model weights (best.pt).
Built With
PyTorch - The core deep learning framework.
Ultralytics YOLOv8 - For the object detection model and pipeline.
Scikit-learn - For data splitting and evaluation metrics.
Pandas - For data manipulation.
Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.
Versioning
We use Semantic Versioning for versioning. For the versions available, see the tags on this repository.
Authors
Your Name - Initial Work - YourGitHubProfile
See also the list of contributors who participated in this project.
License
This project is licensed under the MIT License - see the LICENSE.md file for details.
Acknowledgments
Hat tip to the creators of the TBX11K Dataset.
Inspiration from the broader medical AI research community.
