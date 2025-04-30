Link to model weights, to be put in the same directory as the code: https://drive.google.com/drive/folders/1MSt1eAv62TjxrZejEVjSY4AfUehJs0ds?usp=sharing

Packages to install: !pip install torch torchvision rasterio joblib scikit-learn opencv-python albumentations sickit-learn


To run the inference.py
  With the Unet: python run_inference.py --model unet
  With the RF: python run_inference.py --model rf   
