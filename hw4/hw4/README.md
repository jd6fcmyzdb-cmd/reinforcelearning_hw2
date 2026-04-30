## Setup

### Option 1: Docker

You can use the following Docker image, which already includes all necessary packages for this homework:

```
jellyho/cas4160:0307
```

---

### Option 2: Conda Environment

First, install system dependencies:

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update

apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libgl1-mesa-dev \
    libglew-dev \
    patchelf \
    ffmpeg
```

Then create and activate a conda environment:

```bash
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
```

Move to the hw4/ folder and install Python dependencies:

```bash
pip install swig
pip install -r requirements.txt
pip install -e .
```

---

## TODOs

The TODOs and implementation details are provided in the homework PDF.