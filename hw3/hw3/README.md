## Setup

1. Docker image: `jellyho/cas4160:0307`  
   You can use this Docker image, which already includes all necessary packages for this homework.

2. Conda  
   Start with your `cas4160` conda environment, or create a new one following the instructions from Homework 1.  

   Then, run the following commands under the `hw3/` folder:

    ```
    conda activate [your_env]
    conda install -c conda-forge swig=4.0.2
    pip install -r requirements.txt
    pip install -e .
    ```

## TODOs

The TODOs and implementation details are provided in the homework PDF.