# DATATHON 2025 - Retail Behaviour Analysis

## Overview
Sistem analitik berbasis AI untuk mengoptimalkan tata letak retail melalui analisis perilaku pelanggan dari rekaman CCTV. Menggunakan computer vision dan multimodal transformer untuk memberikan insights actionable bagi strategi bisnis retail.

## 🚀 Key Features
- **Segmentasi Rak**: YOLOv11-seg (94.73% precision, 73.37% recall)
- **Pelacakan Multi-Person**: ByteTrack untuk tracking real-time
- **Klasifikasi Aksi**: TimeSFormer dengan domain adaptation (75.0% accuracy)
- **Analisis Behavior**: Heat-map, dwell-time, dan traffic analysis
- **6 Action Classes**: Reach/Retract from Shelf, Hand in Shelf, Inspect Product/Shelf, Background

## 📊 Outputs
- Heat-map visualisasi traffic pelanggan
- Statistik interaksi per rak
- Rekomendasi optimasi layout toko
- Comprehensive behavioral insights

## Datasets HuggingFace
- MERL Dataset: https://huggingface.co/datasets/haipradana/merl-shopping-action-detection
- CCTV-Like Dataset: https://huggingface.co/datasets/haipradana/action
- Shelf Segmentation: https://huggingface.co/datasets/cheesecz/shelf-segmentation-train

## Models HuggingFace
- Action Recognition Model - Domain Adaptation: https://huggingface.co/haipradana/s-h-o-p-domain-adaptation
- Shelf Segmentation: https://huggingface.co/cheesecz/shelf-segmentation
- YOLOv11 Model: https://huggingface.co/cheesecz/object-tracking

## 🚀🤗 Deployment
This full pipeline has deployed on Huggingface Space: https://huggingface.co/spaces/haipradana/retail-behavior-analysis

### 📹 Demo Video

[Lihat Demo Video](https://github.com/haipradana/DATATHON-2025-Retail-Behaviour-Analysis/blob/main/demo.mp4)
<a href="https://youtu.be/ZtWqnMJQmu0" target="_blank">
  <img src="https://github.com/haipradana/DATATHON-2025-Retail-Behaviour-Analysis/blob/main/demo-screenshot-half.png?raw=true" width="50%">
</a>

