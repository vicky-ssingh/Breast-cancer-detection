# 👤 Face Verification & Identification with Siamese Networks + MobileNetV2

A deep learning project that learns to **verify faces** ("Are these the same person?") and **identify faces** from a registered gallery — built using a Siamese Network with Contrastive Loss and MobileNetV2 transfer learning.

Unlike traditional classification (which needs retraining for new people), this system uses **metric learning**: faces are mapped into a 256-D embedding space where the same person's images cluster together and different people's images are pushed apart.

---

## 🎯 Objectives

- Learn a **face similarity function** using Siamese Networks with Contrastive Loss (more stable than Triplet Loss)
- Use **MobileNetV2** transfer learning as the backbone — efficient, fast, and deployable on edge devices
- Build **two-phase training**: Phase 1 (frozen backbone, train embedding head) → Phase 2 (selective fine-tuning)
- Evaluate with a full **threshold sweep** across 200 operating points, ROC curves, and the official **LFW benchmark**
- Visualize the 256-D embedding space using **UMAP** (better global structure preservation than t-SNE)
- Implement a **face gallery system** with open-set identification and "Unknown" rejection
- Generate a **live verification demo** showing model decisions on individual pairs

---

## 📊 Dataset

**Name:** Labeled Faces in the Wild (LFW)  
**Source:** Built into scikit-learn — `fetch_lfw_people()` and `fetch_lfw_pairs()`  
**Description:** A benchmark dataset of ~13,000 face images of 5,749 public figures collected from news photographs. Images are geometrically aligned ("funneled") so that eyes appear in roughly consistent positions.

| Setting | Value |
|---|---|
| Minimum faces per person | 15 (ensures enough pairs per identity) |
| Official benchmark pairs | ~3,000 predefined same/different pairs |
| Image format | Color (RGB), resized to 96×96 |
| Preprocessing | MobileNetV2 normalization (pixel values → [-1, 1]) |

**Why LFW?**
- Standard academic benchmark — results are comparable across papers
- No Kaggle account or dataset download required (built into scikit-learn)
- Aligned faces reduce pose variation, letting the model focus on identity features

---

## 🛠️ Libraries & Dependencies

Install all required packages:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib seaborn tqdm umap-learn
```

| Library | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.10+ | Neural network training (Keras API) |
| `opencv-python` | any | Image resizing, display conversion |
| `scikit-learn` | any | LFW dataset, ROC-AUC, confusion matrix |
| `matplotlib` | any | Plotting training curves, pair grids |
| `seaborn` | any | Distance heatmaps |
| `tqdm` | any | Progress bars for embedding extraction |
| `umap-learn` | any | UMAP dimensionality reduction (replaces t-SNE) |

**Python version:** 3.8+  
**Recommended:** Run on GPU (Google Colab or local GPU) — training is significantly faster

---

## ▶️ How to Run

### Step 1 — Open the notebook
```bash
jupyter notebook Face_Verification_Siamese_MobileNetV2.ipynb
```
Or upload to **Google Colab** and set Runtime → Change runtime type → **GPU** (T4 is sufficient).

### Step 2 — Run all cells in order
```
Runtime → Run all   (Colab)
Kernel → Restart & Run All   (Jupyter)
```

### Step 3 — What happens automatically
The notebook handles everything end-to-end:

| Step | What Happens |
|---|---|
| Data loading | Downloads LFW via scikit-learn (no Kaggle needed) |
| EDA | Sample face grid, identity distribution, same/different pair visualization |
| Pair generation | 8,000 training pairs (balanced same/different) |
| Model building | MobileNetV2 backbone + custom 256-D embedding head |
| Phase 1 training | Trains embedding head only (backbone frozen) |
| Phase 2 training | Fine-tunes last 30 MobileNetV2 layers |
| Evaluation | Threshold sweep (200 thresholds) + LFW benchmark + ROC |
| UMAP | 2D visualization of 256-D embedding space |
| Gallery demo | Registers 8 people, runs identification queries |
| Live demo | Shows 8 random pairs with model decisions |
| Saving | Saves embedding model + gallery + deployment config |

**Estimated runtime:** ~30–60 minutes on GPU | ~90–150 minutes on CPU

---

## 📈 Key Findings & Model Results

### Architecture

```
[Face Image 96×96] 
       ↓
[MobileNetV2 Backbone — ImageNet pretrained]
       ↓
[Global Average Pooling → 1280-D]
       ↓
[Dense(512, ReLU) + BatchNorm + Dropout(0.4)]
       ↓
[Dense(256)]
       ↓
[L2 Normalize → 256-D Unit-Sphere Embedding]
```

Both branches of the Siamese network share identical weights.

### Loss Function

```
L(y, D) = y · D²  +  (1-y) · max(1.0 - D, 0)²

y = 1 (same person) → minimize D²
y = 0 (different person) → push D beyond margin 1.0
```

### Expected Performance (varies by training run)

| Metric | Approximate Range |
|---|---|
| Verification accuracy (best threshold) | 75–88% |
| Official LFW ROC-AUC | 0.78–0.90 |
| Gallery identification accuracy | 65–82% |
| Optimal distance threshold | 0.6–1.0 |

*Exact values appear in notebook output. Results depend on random initialization and dataset split.*

### Key Findings
- **Contrastive loss** trained more stably than triplet loss with no "collapse" episodes
- **UMAP** visualization showed meaningful clustering — same-person images grouped spatially
- **Threshold sweep** found an optimal operating point ~25% better than the naive default threshold
- **Two-phase training** improved performance: Phase 2 fine-tuning consistently added 3–8% accuracy
- **Distance histograms** showed clear separation between same/different pair distributions

---

## 📁 File Structure

```
project/
│
├── Face_Verification_Siamese_MobileNetV2.ipynb  ← Main notebook
├── README.md                                     ← This file
│
├── face_embedding_mobilenetv2.keras              ← Trained embedding network
├── siamese_contrastive_network.keras             ← Full Siamese model
├── face_gallery.pkl                              ← Registered identity embeddings
├── deployment_config.pkl                         ← Full deployment package
│
└── output_plots/
    ├── lfw_sample_faces.png                      ← Sample face grid
    ├── identity_distribution.png                 ← Class distribution analysis
    ├── same_vs_different_pairs.png               ← Training objective visualization
    ├── training_curves_combined.png              ← Phase 1 + Phase 2 learning curves
    ├── verification_analysis.png                 ← Threshold sweep + ROC + distance dist
    ├── umap_embedding_space.png                  ← 2D UMAP embedding visualization
    ├── embedding_distance_heatmap.png            ← Inter-identity distance matrix
    ├── gallery_identification_results.png        ← Per-person identification accuracy
    └── live_verification_demo.png                ← Model predictions on 8 random pairs
```

---

## 🔄 Key Differences from Classic FaceNet

| Feature | Classic FaceNet | This Notebook |
|---|---|---|
| Loss function | Triplet Loss | **Contrastive Loss** (more stable) |
| Backbone | Custom Inception-ResNet | **MobileNetV2** (efficient, deployable) |
| Embedding size | 128-D | **256-D** (richer representation) |
| Training strategy | End-to-end from scratch | **Two-phase transfer learning** |
| Visualization | t-SNE | **UMAP** (better structure preservation) |
| Threshold selection | Fixed | **200-point sweep** with optimal selection |
| Evaluation | Single pair metric | **Threshold sweep + official LFW benchmark** |
| Gallery system | Basic | **Open-set with "Unknown" rejection** |

---

## ⚠️ Important Disclaimer

This is an **educational project**. The trained models have not been validated for production deployment and must **not** be used for real-world identity verification without:
- Validation on diverse, independent datasets (to check for demographic bias)
- Regulatory compliance assessment
- Privacy impact assessment (GDPR, CCPA, etc.)
- Informed consent from all enrolled individuals

Facial recognition technology carries significant ethical and legal responsibilities.

---

## 🚀 Future Improvements

- **ArcFace loss:** Adds angular margin — state-of-the-art for face recognition (replaces contrastive)
- **Hard negative mining:** Actively find the hardest negative pairs during training for faster convergence
- **VGGFace2 training:** 3.3M images across 9,000 identities — production-quality embeddings
- **Real-time webcam system:** OpenCV video loop + MediaPipe face detection + gallery lookup
- **TFLite deployment:** INT8 quantization for mobile/edge deployment with 4× latency reduction
- **Demographic fairness audit:** Measure and correct for performance disparities across protected groups

---

## 👨‍💻 Author

**VICKY SINGH**


