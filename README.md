# House Price Prediction using Satellite Images and Tabular Data
## 1. Project Overview
### This project aims to predict house prices by combining tabular housing attributes (such as bedrooms, bathrooms,  square footage, etc.) with satellite imagery of the surrounding neighborhood.The motivation is to evaluate whether visual context (green cover, road structure, neighborhood layout) provides complementary information beyond traditional tabular features.This project compares Tabular-only model vs Tabular+Image based fusion model.
## 2. Repository Structure
```text
house-price-prediction-satellite/
│
├── preprocessing_.ipynb        # Data cleaning, EDA, image–price mapping
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
### This downloads all the train and test images.

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
## 7. Results

| Model | R² | RMSE |
|-------|-----|------|
| XGBoost (tabular only) | 0.897 | ~$115,016 |
| CNN (image only) | 0.448 | ~$266,803 |
| XGB + CNN fusion (late) | 0.898 | ~$114,902 |

**Honest finding — satellite imagery did not add predictive value.** The
prediction-level (late) fusion model learned coefficients `[XGB = 1.001,
CNN = 0.000]` — i.e. it assigned **zero weight to the image branch**, making the
"multimodal" model mathematically identical to the tabular-only XGBoost. Even
when allowed to fit in-sample, the image model contributed nothing beyond the
tabular features (location, sqft, grade, etc. already capture most of the
neighbourhood signal that imagery would provide). This is a clean **negative
result**: on this data, tabular features dominate and late fusion does not help.

## 8. Limitations & Future Work
- The fusion above is **late (prediction-level) fusion**, evaluated **in-sample**
  for the stacker, so its tiny lift is optimistic — but the zero-weight result
  holds regardless.
- A fairer test of whether imagery helps is **feature-level fusion**: concatenate
  the CNN's embedding with the tabular features and train a single model, which
  can capture image signal that final-price stacking cannot.
- Satellite images are limited in resolution due to Google Maps API constraints;
  higher-resolution / multi-zoom imagery plus explicit geospatial features
  (greenery, road density, POIs) would be the next thing to try.

---
**Author:** Mayank Kumar Agrawal, IIT Roorkee