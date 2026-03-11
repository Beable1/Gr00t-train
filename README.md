## GR00T Training on `Beable/dexterous_ee_56`

This folder contains a minimal setup to **fine‑tune NVIDIA Isaac GR00T (via LeRobot)** on your existing dataset `Beable/dexterous_ee_56`, intended to be run on a cloud GPU machine.

The goal is to:
- Use **`policy.type=groot`** instead of plain diffusion.
- **Avoid the previous issues** (tiny 84×84 crops, mis‑configured vision) by **disabling image cropping** and using the full 360×640 frames.
- Make it easy to run both **single‑GPU** and **multi‑GPU** training with one script.

---

### 1. Create environment on the cloud machine

You can use either `conda` or a bare Python environment. Example with conda:

```bash
conda create -y -n groot python=3.10
conda activate groot

# System ffmpeg (recommended, needed for video)
conda install -y -c conda-forge ffmpeg=7.1.1
```

---

### 2. Install Python dependencies

From the project root (where this `GR00T` folder lives):

```bash
cd GR00T
pip install --upgrade pip

# 1) Install PyTorch + torchvision for your CUDA version
#    (See: https://pytorch.org/get-started/locally/)
pip install "torch>=2.2.1,<2.8.0" "torchvision>=0.21.0,<0.23.0"

# 2) Install the rest from requirements.txt
pip install -r requirements.txt
```

Notes:
- `flash-attn` may require a recent GPU and CUDA. If pip fails to build it,
  you can comment it out from `requirements.txt` and install an officially
  provided wheel manually following the FlashAttention docs.
- `lerobot[groot]>=0.4.0` installs LeRobot with GR00T support and the CLI
  tools (`lerobot-train`, `lerobot-eval`, etc.).

---

### 3. Run GR00T training on your dataset

Single‑GPU training (recommended first test):

```bash
cd /path/to/your/repo   # project root
conda activate groot

python GR00T/train_groot_beable.py \
  --output_dir ./outputs/groot_beable \
  --steps 40000 \
  --batch_size 32
```

Multi‑GPU training (if the cloud machine has multiple GPUs and `accelerate` is configured):

```bash
python GR00T/train_groot_beable.py \
  --output_dir ./outputs/groot_beable \
  --steps 40000 \
  --batch_size 64 \
  --num_gpus 4
```

The script internally launches:

- `policy.type=groot`  → GR00T policy head
- `policy.tune_diffusion_model=false`  → only the action head is fine‑tuned,
  which is safer for a relatively small custom dataset.
- `dataset.repo_id=Beable/dexterous_ee_56`
- `dataset.use_imagenet_stats=true`, `dataset.video_backend=torchcodec`
- `policy.crop_shape=null`  → **no random/center crop**, the model always
  sees the full 360×640 images (this avoids the previous 84×84 crop issue).

---

### 4. Hugging Face access

If `Beable/dexterous_ee_56` is **private**, make sure the machine has a valid
Hugging Face token:

```bash
huggingface-cli login
```

If it is public, no extra authentication is needed.

---

### 5. Outputs

Checkpoints and logs will be written under the `--output_dir` you provide
(default: `./outputs/groot_beable`). You can later use these checkpoints for
deployment/inference similarly to your previous diffusion policy setup.

