import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import h5py
from PIL import Image
import io
import joblib


def checksampling(df):
    dist = df['target'].value_counts()
    return dist

def downsampling(df, num = 1):
    malignant = df[df['target'] == 1]
    benign = df[df['target'] == 0].sample(n = len(malignant)*num, random_state = 42)
    downsample_df = pd.concat([malignant, benign])
    return downsample_df



if __name__ == '__main__':
    df = pd.read_csv('Desktop/UCD/Fall_2024/STA_221/Vision/data/train-metadata.csv')
    downsample_df = downsampling(df, num = 3)
    dist = checksampling(downsample_df)
    # print(dist)
    # save data in file
    joblib.dump(downsample_df, '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/data/downsamples.csv')
    print(downsample_df)

