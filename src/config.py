# -------------------------
# 📁 Paths
# -------------------------
DATASET_PATH = "data/celebv-text"                                   # Root directory of the CelebV dataset
MOTION_GENERATION_MODEL_PATH = "model/text_2_motion_model.pt"       # Path to save/load motion generation model
OUTPUT_VIDEO_PATH = "output/video/video.mp4"                        # Output path for rendered video
FLAME_HEAD_TEMPLATE_PATH = "assets/flame_head_template.obj"         # Path to FLAME 3D head template


# -------------------------
# 📊 Data & Model Dimensions
# -------------------------
FRAMES_PER_VIDEO = 32                               # Number of frames to generate per video
MOTION_FEATURES_PER_FRAME = 50                      # Number of motion features predicted per frame
TEXT_EMBEDDING_DIM = 512                            # Text embedding dimension (CLIP=512, BERT=768)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"    # Pretrained CLIP model name


# -------------------------
# ⚙️ Training Hyperparameters
# -------------------------
LEARNING_RATE = 1e-4                # Initial learning rate for optimizer
WEIGHT_DECAY = 1e-5                 # L2 regularization term
BATCH_SIZE = 2                      # Number of samples per training batch
NUM_EPOCHS = 2                      # Number of epochs for model training


# -------------------------
# 🎬 Video Rendering Settings
# -------------------------
FPS = 25                            # Frames per second for rendered video
IMG_SIZE = 256                      # Frame resolution (width and height in pixels)
