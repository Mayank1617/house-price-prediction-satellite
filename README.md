# House Price Prediction using Satellite Images and Tabular Data
## 1. Project Overview
### This project aims to predict house prices by combining tabular housing attributes (such as bedrooms, bathrooms,  square footage, etc.) with satellite imagery of the surrounding neighborhood.The motivation is to evaluate whether visual context (green cover, road structure, neighborhood layout) provides complementary information beyond traditional tabular features.This project compares Tabular-only model vs Tabular+Image based fusion model.
## 2. Repository Structure
```text
house-price-prediction-satellite/
│
├── preprocessing.ipynb        # Data cleaning, EDA, image–price mapping
├── model_training.ipynb       # XGBoost, CNN training and fusion model
├── data_fetcher.py            # Script to download satellite images via API
├── 23323021_final.csv         # Final test predictions (submission file)
├── 23323021_report.pdf        # Project report (PDF)
├── README.md                  # Project documentation
└── LICENSE
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
### This downloads the training and test images.

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
