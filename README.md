# Low-Dose CT Denoising Research

This repository contains implementations and experiments for various low-dose CT (LDCT) denoising approaches, focusing on deep learning-based methods with anatomical awareness.

## Repository Structure

```
📦 Repository
├── 📁 redcnn_v1/                 # Baseline comparison experiments
│   ├── RED-CNN implementation
│   ├── U-Net implementation
│   └── Performance comparison with various modifications
├── 📁 redcnn_v2(aide)/          # A-IDE framework implementation
│   └── Agent-Integrated Denoising Experts
├── 📁 redcnn_v3(bioatt)/        # BioAtt framework implementation
│   └── Anatomical Prior Driven Denoising
├── 📁 utils/                    # Utility functions and helper modules
├── 📄 .DS_Store
└── 📄 README.md
```

## Project Overview

### 📁 redcnn_v1 - Baseline Experiments
This folder contains comprehensive comparisons between RED-CNN and U-Net architectures for LDCT denoising. Various modifications were tested to optimize performance:

- **VGG Loss Integration**: Added perceptual loss for better feature preservation
- **Batch Size Variations**: Experimented with different batch sizes for optimal training
- **Activation Function Changes**: Tested various activation functions for improved convergence
- **Performance Metrics**: Comparative analysis using RMSE, PSNR, and SSIM

### 📁 redcnn_v2(aide) - Agent-Integrated Denoising Experts

Implementation of the **A-IDE (Agent-Integrated Denoising Experts)** framework:

> **Paper**: "A-IDE: Agent-Integrated Denoising Experts"  
> **Authors**: Uihyun Cho, Namhun Kim  
> **arXiv**: https://arxiv.org/abs/2503.16780

**Key Features:**
- **Multi-Expert Architecture**: Three anatomical region-specialized RED-CNN models
- **LLM Agent Management**: Decision-making agent for dynamic model routing
- **Semantic Analysis**: BiomedCLIP integration for anatomical understanding
- **Automatic Task Distribution**: Prevents overfitting through intelligent load balancing

**Advantages:**
- Excels in heterogeneous, data-scarce environments
- Eliminates manual intervention through automated pipeline
- Superior performance on Mayo-2016 dataset

### 📁 redcnn_v3(bioatt) - Anatomical Prior Driven Denoising

Implementation of the **BioAtt** framework:

> **Paper**: "BioAtt: Anatomical Prior Driven Low-Dose CT Denoising"  
> **Authors**: Namhun Kim, UiHyun Cho  
> **arXiv**: https://arxiv.org/abs/2504.01662

**Key Innovations:**
- **Anatomical Prior Integration**: Leverages BiomedCLIP for anatomical guidance
- **Structure-Preserving Attention**: Focuses on clinically relevant anatomical regions
- **Novel Architecture**: Embeds anatomic priors directly into spatial attention mechanisms

**Main Contributions:**
- Outperforms baseline and attention-based models across multiple metrics
- Introduces new paradigm for anatomical prior integration
- Provides interpretable attention maps for clinical validation

## Performance Evaluation

All models are evaluated using standard medical image quality metrics:
- **RMSE** (Root Mean Square Error)
- **PSNR** (Peak Signal-to-Noise Ratio)  
- **SSIM** (Structural Similarity Index)

## Dataset

Experiments are conducted on the **Mayo-2016 dataset**, a standard benchmark for low-dose CT denoising research.

## Getting Started

```bash
# Clone the repository
git clone [repository-url]

# Navigate to specific implementation
cd redcnn_v1    # For baseline experiments
cd redcnn_v2    # For A-IDE framework
cd redcnn_v3    # For BioAtt framework
```

## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{cho2025aide,
  title={A-IDE: Agent-Integrated Denoising Experts},
  author={Cho, Uihyun and Kim, Namhun},
  journal={arXiv preprint arXiv:2503.16780},
  year={2025}
}

@article{kim2025bioatt,
  title={BioAtt: Anatomical Prior Driven Low-Dose CT Denoising},
  author={Kim, Namhun and Cho, UiHyun},
  journal={arXiv preprint arXiv:2504.01662},
  year={2025}
}
```

## Contact

For questions or collaborations, please reach out to the authors.
