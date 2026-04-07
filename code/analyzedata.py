#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_data.py
PathMNIST Dataset Analysis Script
Loads the PathMNIST dataset, displays statistics, visualizes samples,
and shows class distribution across splits.
"""

import numpy as np # import numpy with the alians np
import matplotlib.pyplot as plt #import pyplot with the alias plt
import matplotlib.gridspec as gridspec #import gridspec with the alias gridspec
from os.path import join #import join

# ─────────────────────────────────────────────
# 1. Load the dataset
# ─────────────────────────────────────────────

data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedMNIST/" # save file path with variable
data = np.load(join(data_dir, "pathmnist.npz")) #use the load method in mp class to open npz file, the join function constructructs the full path of the file

train_images = data["train_images"]   # accesses the train_images section of the numpy object with the npz file shape: (N, 28, 28, 3)
train_labels = data["train_labels"]   # accesses the train_labels section of the numpy object with the npz file shape: (N, 1)
val_images   = data["val_images"]       # accesses the val_images section of the numpy object with the npz file
val_labels   = data["val_labels"]       # accesses the val_labels section of the numpy object with the npz file
test_images  = data["test_images"]      # accesses the test_images section of the numpy object with the npz file
test_labels  = data["test_labels"]      # accesses the test_labels section of the numpy object with the npz file

# Flatten label arrays to 1-D
train_labels = train_labels.flatten()   # use the flatten method to put the training labels array to a 1D vector
val_labels   = val_labels.flatten()     # use the flatten method to put the value labes array into a 1D vector
test_labels  = test_labels.flatten()    # use the flatten method to put the test labels array to a 1D vector

# PathMNIST has 9 tissue-type classes
CLASS_NAMES = [
    "Adipose",          # 0
    "Background",       # 1
    "Debris",           # 2
    "Lymphocytes",      # 3
    "Mucus",            # 4
    "Smooth Muscle",    # 5
    "Normal Colon",     # 6
    "Cancer Stroma",    # 7
    "Colorectal Adenocarcinoma",  # 8
]
NUM_CLASSES = len(CLASS_NAMES) #uses len method to get length of class_names variable

# ─────────────────────────────────────────────
# 2. Basic statistics
# ─────────────────────────────────────────────

print("=" * 60) # basic print
print("PathMNIST Dataset Summary") #basic print
print("=" * 60) #basic print
print(f"Image shape : {train_images.shape[1:]}  (H x W x C)") # array structure train_images is shape[0] number of images and shape[1:] is the height width and channel, uses fstring print to print out image shape
print(f"Dtype       : {train_images.dtype}") #f string prints the data tyep fo train_images
print(f"Pixel range : [{train_images.min()}, {train_images.max()}]") # f string print min and max pixel value in train_images
print() #newline
print(f"{'Split':<12} {'# Images':>10}") #fstring print the word split left aligned 12 and images right alligned 10
print("-" * 24)
print(f"{'Train':<12} {len(train_images):>10,}") # fstring prints number of training images
print(f"{'Validation':<12} {len(val_images):>10,}") # fstring prints validation left aligned 12 and val_images variable right aligned 10 
print(f"{'Test':<12} {len(test_images):>10,}") # fstring print the word Test left aligned 12 and length of test_images right aligned 10
print(f"{'Total':<12} {len(train_images)+len(val_images)+len(test_images):>10,}") # fstring print the word total left aligned by 12 length of train images and val_images and test_images right aligned by 10
print() #new line

print(f"{'Class':<32} {'Train':>8} {'Val':>8} {'Test':>8}") #f string print the word class left aligned 32 train right aligned 8 val right aligned 8 test right aligned 8
print("-" * 60) #basic print
for c, name in enumerate(CLASS_NAMES): #enumerates through class name taking in c (the index) and name(the class name)
    tr = np.sum(train_labels == c) #counts number of training labels in CLASS_NAMES that equal the training index c
    v  = np.sum(val_labels   == c) #counts number of value labels in CLASS_NAMES that equal the training index c
    te = np.sum(test_labels  == c) #counts number of test labels in CLASS_NAMES that equal the training index c
    print(f"{name:<32} {tr:>8,} {v:>8,} {te:>8,}") # print the variables name left aligned 32 tr right aligned 8 v right aligned 8 and te right aligned 8
print()

# ─────────────────────────────────────────────
# 3. Visualize random samples from each class
# ─────────────────────────────────────────────

SAMPLES_PER_CLASS = 5 #global variable samples per class assigned to 5
np.random.seed(42) = #generate random number from numpy with seed 42

fig, axes = plt.subplots(
    NUM_CLASSES, SAMPLES_PER_CLASS,
    figsize=(SAMPLES_PER_CLASS * 1.8, NUM_CLASSES * 1.8)
) # setting size of matplotlib figure, width of 1.8 * samples per class(5) height of 1.8 times number of classes (9) 
fig.suptitle("PathMNIST – Random Samples per Class (Training Set)", #assign a title to the figure 
             fontsize=13, fontweight="bold", y=1.01)                # with a font size of 13 bolded positioned 1.01 inches above the figure

for row, c in enumerate(range(NUM_CLASSES)):#enumerate through the size of NUM_CLASSES with variable row and c
    idxs = np.where(train_labels == c)[0] # assigns index variable to the indices of all training samples in c
    chosen = np.random.choice(idxs, SAMPLES_PER_CLASS, replace=False)
    for col, idx in enumerate(chosen):
        ax = axes[row, col]
        ax.imshow(train_images[idx])
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(CLASS_NAMES[c], fontsize=7, rotation=0,
                          labelpad=60, va="center")

plt.tight_layout()
plt.savefig("samples_per_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: samples_per_class.png")

# ─────────────────────────────────────────────
# 4. Class distribution bar charts
# ─────────────────────────────────────────────

splits = {
    "Train":      train_labels,
    "Validation": val_labels,
    "Test":       test_labels,
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle("Class Distribution Across Splits", fontsize=13,
             fontweight="bold")

colors = plt.cm.tab10(np.linspace(0, 0.9, NUM_CLASSES))

for ax, (split_name, labels) in zip(axes, splits.items()):
    counts = [np.sum(labels == c) for c in range(NUM_CLASSES)]
    bars = ax.bar(range(NUM_CLASSES), counts, color=colors)
    ax.set_title(f"{split_name}  (n={len(labels):,})", fontsize=11)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Class")
    # Annotate counts on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}", ha="center", va="bottom", fontsize=6
        )

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: class_distribution.png")

# ─────────────────────────────────────────────
# 5. Per-split summary table (printed)
# ─────────────────────────────────────────────

print("=" * 60)
print("Per-Class Image Counts")
print("=" * 60)
header = f"{'Class':<32} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}"
print(header)
print("-" * len(header))
for c, name in enumerate(CLASS_NAMES):
    tr = int(np.sum(train_labels == c))
    v  = int(np.sum(val_labels   == c))
    te = int(np.sum(test_labels  == c))
    print(f"{name:<32} {tr:>8,} {v:>8,} {te:>8,} {tr+v+te:>8,}")

print("=" * 60)
print("Analysis complete.")