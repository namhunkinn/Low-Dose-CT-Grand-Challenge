# **🌟 Low-Dose CT Grand Challenge: REDCNN and UNet with Attention Mechanisms**

## **📌 Introduction**
This project focuses on enhancing **low-dose CT (LDCT)** images into high-quality **normal-dose CT (NDCT)** images, improving diagnostic efficiency while minimizing radiation exposure.  

### **Key Details**:
- **Task**: Image Denoising  
- **Models**:  
  - **REDCNN**: Residual Encoder-Decoder CNN for low-level feature enhancement  
  - **UNet**: Multi-level feature extraction with encoder-decoder architecture  
  - **Attention Mechanisms**:  
    - Channel Attention  
    - Spatial Attention  

---

## **⚙️ Data Preprocessing**
1. **Normalization**:
   - Pixel values scaled from **(-1000, 2000)** to **(-1, 2)**.  
2. **Patch Division**:
   - Original images (**512×512**) divided into **81 patches** of size **55×55**.  
3. **Cosine Similarity**:
   - ResNet50 extracted features to compute cosine similarity (~**0.996**) between LDCT and NDCT patches, ensuring consistency.

---

## **🏗️ Models**

### **1. REDCNN**  
- **Features**:
  - Fixed **channel size**: 96  
  - Residual connections for enhancing **low-level features** (e.g., edges, textures).  
  - Maintains spatial resolution, focusing on detailed reconstruction.  

- **Improvements**:
  - Batch Normalization  
  - Kaiming He Initialization  
  - **VGG16 Loss**: Improves perceptual quality by focusing on image realism.  

---

### **2. UNet**
- **Features**:
  - Encoder-decoder architecture with skip connections for **multi-level feature learning**.  
  - Dynamically adjusts spatial and channel dimensions.  

- **Enhancements with Attention**:
  - **Channel Attention**: Amplifies key feature channels.  
  - **Spatial Attention**: Highlights critical spatial regions for better feature localization.

---

## **📊 Evaluation Metrics**
| **Metric** | **Description** | **Goal** |
|------------|-----------------|----------|
| **RMSE**   | Measures pixel-wise error. | Lower is better. |
| **PSNR**   | Quantifies signal-to-noise ratio. | Higher is better. |
| **SSIM**   | Evaluates structural similarity. | Closer to 1 is better. |

---

## **🔬 Experimental Results**
### **Performance Summary**:
- **Base Models**:
  - **REDCNN**: Superior in **RMSE** and **SSIM**, excelling in low-level detail preservation.  
  - **UNet**: Better **PSNR** performance, leveraging comprehensive multi-level abstractions.  
- **With Attention Mechanisms**:
  - Significant improvement observed in both REDCNN and UNet by enhancing feature importance through **Channel** and **Spatial Attention**.

---

## **📌 Conclusion**
- **REDCNN** is highly effective for **low-level feature tasks** (e.g., edge and texture preservation).  
- **UNet** provides robust performance across **multi-level feature reconstruction** tasks.  
- Integrating **Channel and Spatial Attention** mechanisms leads to substantial performance gains, refining both spatial and feature-level prioritization.

---

## **📚 References**
1. **Chen, H., et al.** (2017). *Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)*. *IEEE Transactions on Medical Imaging*.  
2. **Ronneberger, O., et al.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. *MICCAI*.  

🔗 **GitHub Repository**: [Low-Dose CT Grand Challenge](https://github.com/ksouth0413/Low-Dose-CT-Grand-Challenge)
