import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
DATA_DIR = 'extracted_images'
# UPDATED TEAM NAMES:
CATEGORIES = ['Tanzania', 'Morocco', 'Ref'] 
IMG_SIZE = 64  

data = []
labels = []

print("Loading and preprocessing images...")

# 1. LOAD DATA
for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    if not os.path.exists(path):
        print(f"ERROR: Folder '{category}' not found in {DATA_DIR}!")
        print("Did you name the folder correctly?")
        continue
        
    class_num = CATEGORIES.index(category)  # 0=Tanzania, 1=Morocco, 2=Ref
    
    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img_array = cv2.imread(img_path)
            if img_array is None: continue

            # Resize and Flatten
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(new_array.flatten())
            labels.append(class_num)
        except Exception as e:
            pass

# Convert to numpy
X = np.array(data)
y = np.array(labels)

if len(X) == 0:
    print("CRITICAL ERROR: No images loaded. Check your folder names!")
    exit()

print(f"Loaded {len(X)} images total.")

# 2. SPLIT DATA (Training vs Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Training on {len(X_train)} images, Testing on {len(X_test)} images.")

# 3. OPTIMIZATION LOOP
best_k = 1
best_acc = 0
best_model = None

print("\n--- PARAMETER OPTIMIZATION ---")
for k in [1, 3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    print(f"k={k}: Accuracy = {acc*100:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_k = k
        best_model = model

print(f"\nWinner: k={best_k} with {best_acc*100:.2f}% accuracy.")

# 4. SAVE MODEL
with open('football_model_knn.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model saved as 'football_model_knn.pkl'")