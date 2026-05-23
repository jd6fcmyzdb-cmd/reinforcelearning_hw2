## Setup

First, install the system dependencies:

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

Move to the hw6/ folder and install python dependencies:

```bash
conda install -c conda-forge swig=4.0.2
pip install -r requirements.txt
pip install -e .
```

## TODOs

The TODOs and descriptions for the code that needs to be implemented are provided in the homework PDF.