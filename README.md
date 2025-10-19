# 🧠 Text-to-Head-Motion Generation from Text

## 📖 Overview
This project performs **text-to-motion generation**, converting textual descriptions into **3D head motion sequences** and rendering them as **video animations**.  

- ⚙️ **Pretrained text encoder (CLIP)** — converts text prompts into embeddings  
- 🎯 **Motion generation network** — predicts per-frame motion parameters  
- 🎥 **3D rendering using FLAME template** (fallback to 2D dot visualization if unavailable)  
- 📉 **MSE Loss** and **Adam optimizer** for training motion generation  
- 🧱 **Modular architecture** — clear separation of model, training, inference, and rendering  
- 📊 **Real-time loss logging** during training  
- 🌐 Optional **video output** for visualization  

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **Transformers** – CLIP text encoder  
- **numpy**, **scipy** – data handling  
- **torch.utils.data** – dataset and dataloader  
- **Pillow**, **imageio**, **opencv-python** – video and image handling  
- **trimesh**, **pyrender**, **pyglet** – optional 3D rendering  
- **face-alignment** – for FLAME/DECA landmarks  
- **tqdm** – progress bars  

---

## ⚙️ Requirements

- Python **3.15+**
- Recommended editor: **VS Code**

---

## 📦 Installation

- Clone the repository
```bash
git clone https://github.com/hurkanugur/Text2Video.git
```

- Navigate to the `Text2Video` directory
```bash
cd Text2Video
```

- Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Setup Python Environment in VS Code

1. `View → Command Palette → Python: Create Environment`  
2. Choose **Venv** and your **Python version**  
3. Select **requirements.txt** to install dependencies  
4. Click **OK**

---

## 📂 Project Structure

```bash
assets/
└── flame_head_template.png           # Flame head model

data/
└── celebv-text                       # Training datasets

output/
├── mesh                              # Generated mesh file
└── video                             # Generated video file

model/
└── text_2_motion_model.pt            # Trained model

src/
├── config.py                         # Paths, hyperparameters, ...
├── dataset.py                        # Data loading & preprocessing
├── train.py                          # Training pipeline
├── inference.py                      # Inference pipeline
├── model.py                          # Motion generation model (text → motion parameters)
└── text_encoder.py                   # Text encoder model (text → embeddings)

main/
├── main_train.py                     # Entry point for training
└── main_inference.py                 # Entry point for inference

requirements.txt                      # Python dependencies
```

---

## 📂 Model Architecture

```bash
Input Text
    ↓
Pretrained CLIP Text Encoder
    ↓
Text Embedding (feature vector)
    ↓
Text2MotionModel (MLP)
    ↓
Per-frame Motion Parameters (FLAME)
    ↓
3D Rendering (pyrender) or 2D fallback
    ↓
Generated Video Output
```

---

## 📂 Train the Model
Navigate to the project directory:
```bash
cd Text2Video
```

Run the training script:
```bash
python -m main.main_train
```
or
```bash
python3 -m main.main_train
```

---

## 📂 Run Inference / Make Predictions
Navigate to the project directory:
```bash
cd Text2Video
```

Run the app:
```bash
python -m main.main_inference
```
or
```bash
python3 -m main.main_inference
```
