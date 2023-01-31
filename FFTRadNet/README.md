# FFTRadNet
1. Open-source dataset and a multi-task architecture (https://github.com/valeoai/RADIal/tree/main/FFTRadNet) consists of five blocks:

    • A `pre-encoder` reorganizing and compressing the RD tensor into a meaningful and compact representation;
    
    • A shared `Feature Pyramidal Network (FPN)` encoder combining low-resolution semantic information with high-resolution details;
    
    • A `range-angle decoder` building a range-azimuth latent
representation from the feature pyramid;
    
    • A `detection head` localizing vehicles in range-azimuth coordinates;
    
    • A `segmentation head` predicting the free driving space.

2. This folder contains .py files for Gemmini quantization and parameters (weights, bias, scale, zero_point) preparation.

3. The .c and .h files can be found:
    ```bash
    ../imagenet/gemmini_fftradnet_*.c
    ../imagenet/gemmini_fftradnet_*_params.h
    ```