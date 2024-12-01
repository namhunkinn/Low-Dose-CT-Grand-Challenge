### README 작성 내용

---

# **Low-Dose CT Grand Challenge: REDCNN and UNet with Attention Mechanisms**

## **Introduction**
This project aims to enhance **low-dose CT (LDCT)** images by denoising them into high-quality **normal-dose CT (NDCT)** images. The focus lies on improving diagnostic efficiency while minimizing radiation exposure.

- **Task**: Image Denoising  
- **Models**:  
  - **REDCNN**: Low-level feature-focused residual encoder-decoder CNN  
  - **UNet**: Multi-level feature extraction and reconstruction network  
  - **Attention Mechanisms**:  
    - **Channel Attention**  
    - **Spatial Attention**  

---

## **Data Preprocessing**
1. **Normalization**:
   - Pixel values normalized from **(-1000, 2000)** to **(-1, 2)**.
2. **Patch Division**:
   - Original images of size **512×512** split into **81 patches** of **55×55**.
3. **Cosine Similarity**:
   - ResNet50 used for feature extraction; cosine similarity demonstrated high similarity (~0.996) between LDCT and NDCT patches.

---

## **Models**

### **1. REDCNN**  
- **Key Features**:
  - Fixed **channel size**: 96  
  - Residual connections for **low-level feature enhancement** (e.g., edges and textures).  
  - Feature maps maintain similar spatial resolution to the original input, prioritizing **low-level details**.  

- **Improvements**:
  - Batch Normalization  
  - Kaiming He Initialization  
  - Incorporation of **VGG16 Loss** for perceptual enhancement.

### **2. UNet**
- **Key Features**:
  - Encoder-decoder architecture with skip connections.  
  - Progressive feature extraction from **low-level to high-level details**.  
  - Spatial and channel dimensions dynamically adjusted.  

- **Enhancements with Attention**:
  - **Channel Attention** to amplify important channels.  
  - **Spatial Attention** to emphasize critical spatial regions.

---

## **Evaluation Metrics**
1. **RMSE**: Measures pixel-wise error magnitude. (Lower is better.)  
2. **PSNR**: Quantifies signal-to-noise ratio. (Higher is better.)  
3. **SSIM**: Evaluates structural similarity between images. (Closer to 1 is better.)  

---

## **Experimental Results**
- **Performance Comparison**:
  - **Base Models**:
    - REDCNN outperformed UNet on RMSE and SSIM metrics, reflecting its low-level feature specialization.  
    - UNet excelled in PSNR, leveraging its ability to capture multi-level abstractions.
  - **With Attention**:
    - Attention mechanisms significantly enhanced performance for both REDCNN and UNet by refining feature importance.

---

## **Conclusion**
- REDCNN excels in **low-level feature extraction**, making it suitable for tasks prioritizing edges and textures.  
- UNet is ideal for comprehensive feature learning, combining low and high-level details.  
- **Channel and Spatial Attention** modules offer substantial improvements by selectively focusing on critical features.

---

### **References**
1. Chen, H., et al. (2017). *Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)*. IEEE Transactions on Medical Imaging.  
2. Ronneberger, O., et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.  

**GitHub Repository**: [Low-Dose CT Grand Challenge](https://github.com/ksouth0413/Low-Dose-CT-Grand-Challenge)  
