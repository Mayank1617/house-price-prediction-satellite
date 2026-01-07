# House Price Prediction using Satellite Images and Tabular Data
## 1. Project Overview
### This project aims to predict house prices by combining tabular housing attributes (such as bedrooms, bathrooms,  square footage, etc.) with satellite imagery of the surrounding neighborhood.The motivation is to evaluate whether visual context (green cover, road structure, neighborhood layout) provides complementary information beyond traditional tabular features.This project compares Tabular-only model vs Tabular+Image based fusion model.
## 2. Repository Structure
```text
CDC/
├── data_fetcher.py                  # Script to download satellite images via API
├── README.md                        # Project setup and usage instructions
├── 23323021_report.pdf              # Final project report (PDF)

├── data/
│   └── raw/
│       ├── train1.xlsx              # Training tabular data
│       ├── test2.xlsx               # Test tabular data
│       ├── images_train          # Downloaded satellite images (train)
│       └── images_test             # Downloaded satellite images (test)

├── notebooks/
│   ├── preprocessing_.ipynb          # Data cleaning, EDA, image–price mapping
│   ├── model_training.ipynb         # Tabular, CNN, and fusion model training
│   └── 23323021_final.csv            # Final test set predictions (submission file)

├── models/
│   ├── image_only_best.pth           # Best CNN model checkpoint
│   ├── image_only_checkpoint.pth     # Intermediate CNN checkpoint
│   └── training_history.json         # Training metrics and loss curves

 ```
## 3. Setup Instructions
### 3.1 Clone the Repository 
```bash 
git clone https://github.com/Mayank1617/house-price-prediction-satellite.git
cd house-price-prediction-satellite
 ```
### 3.2 Create Python Environment
```bash
 conda create -n satellite_ml python=3.9
 conda activate satellite_ml
```
### 3.3 Install Dependencies
```bash
 pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision pillow tqdm joblib
 ```
## 4. Satellite Image Download
### 4.1 Set API Key
```bash
export GOOGLE_MAPS_API_KEY="YOUR_API_KEY"
```
### 4.2 Download Images
```bash
python data_fetcher.py 
```
### This downloads and stores the images in 
```text
data/raw/images_train
data/raw/images_test
```

## 5. Run the Preprocessing Notebook
```bash
jupyter notebook preprocessing_.ipynb 
```
### This notebook perfroms data cleaning and visualization , maps the image id to prices.
## 6. Run the Training Notebook
```bash
jupyter notebook model_training.ipynb
```
### This notebook trains tabular only XGBoost model , trains a CNN on satellite images, combines predictions using fusion model , evaluates models using R2 and RMSE.
## 7. Notes and Limitations
### --> Satellite Images are limited in resolution due to API constraints.
### --> Larger datasets or higher-resolution imagery may further improve fusion performance.
---
**Author:** Mayank Kumar Agrawal, IIT Roorkee