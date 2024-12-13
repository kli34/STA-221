# STA-221

# Enhanced Skin Cancer Detection Using Machine Learning on 3D Total Body Photography (3D-TBP) Images

Skin cancer can be fatal if not detected early. In the past, traditional methods for identifying benign skin cancer required significant expertise from dermatologists and were time-consuming. This made early detection particularly challenging in underdeveloped areas with limited access to specialists. Recently, advancements in machine learning (ML) and computer vision have shown great potential in automating the detection process, enabling more accurate and timely interventions.

This project focuses on addressing critical challenges such as the variability in lesion appearances across different skin tones and lighting conditions in 3D-TBP images, both of which significantly impact detection accuracy. By developing a reliable ML model, we aim to assist dermatologists in improving diagnostic accuracy and expanding access to care, especially for underserved populations with limited specialist resources

# Dataset

The dataset used in this project is from the Kaggle competition ISIC 2024 Skin Cancer Detection with 3D-TBP. It is accessible via the following link: https://kaggle.com/competitions/isic-2024-challenge. The dataset comprises two components: lesion images as features and corresponding labels as targets, where 0 indicates benign lesions and 1 represents malignant lesions. The dataset is highly imbalanced, containing 393 malignant samples and 1,048,182 benign samples. It is also worth noting that benign lesions tend to be relatively uniform in appearance, whereas malignant lesions vary significantly. For instance, Nevus, as illustrated in the graph below, closely resembles benign lesions but is distinct from the other four types of malignant lesions. This variability among malignant lesions poses additional challenges for accurate classification and highlights the importance of effective sampling strategies.

# Methods

By Kuang Li:
- Logistic Regression
- Naive Bayes
- Vision Transformer

By Zecheng Li:
- XGBoost

By Zhuorui He:
- Support Vector Machine

By Shiyu Wu:
- GAN-Based Oversampling
