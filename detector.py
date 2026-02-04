from tensorflow.keras.models import load_model

import cv2
import numpy as np
import os

def recommend_trees(forest_percent, region="Aravalli"):
    if region.lower() == "aravalli":
        if forest_percent < 15:
            return [
                "Neem (Azadirachta indica)",
                "Babul (Acacia nilotica)",
                "Peepal (Ficus religiosa)",
                "Ber (Ziziphus mauritiana)"
            ]
        elif forest_percent < 40:
            return [
                "Dhok (Anogeissus pendula)",
                "Khejri (Prosopis cineraria)",
                "Arjun (Terminalia arjuna)"
            ]
        else:
            return [
                "Jamun (Syzygium cumini)",
                "Banyan (Ficus benghalensis)",
                "Peepal (Ficus religiosa)"
            ]
    else:
        return ["Local native species recommended"]

MODEL_PATH = "forest_segmentation_model.h5"
ml_model = load_model(MODEL_PATH)
print("ML Model loaded successfully")

def generate_geospatial_map(mask_before, mask_after, output_path):
    h, w = mask_before.shape

    geo_map = np.zeros((h, w, 3), dtype=np.uint8)
    # Background (Non-Forest) - White
    geo_map[:] = [255, 255, 255]

    # Existing Trees (Green) - Where trees are present in 'after' image
    # Covers both Stable Forest and Reforestation
    geo_map[mask_after > 0] = [34, 139, 34]

    # Deforestation (Red) - Where trees were removed
    geo_map[(mask_before > 0) & (mask_after == 0)] = [0, 0, 255]

    cv2.imwrite(output_path, geo_map)


def analyze_forest_change(before_path, after_path, output_dir="static/outputs"):

    os.makedirs(output_dir, exist_ok=True)

    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    before = cv2.resize(before, (512, 512))
    after = cv2.resize(after, (512, 512))

    before_hsv = cv2.cvtColor(before, cv2.COLOR_BGR2HSV)
    after_hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask_before = cv2.inRange(before_hsv, lower_green, upper_green)
    mask_after = cv2.inRange(after_hsv, lower_green, upper_green)

    forest_before = np.sum(mask_before > 0)
    forest_after = np.sum(mask_after > 0)
    total_pixels = mask_before.size

    percent_before = (forest_before / total_pixels) * 100
    percent_after = (forest_after / total_pixels) * 100
    change = percent_after - percent_before

    if change > 0:
        result = "Afforestation 🌱"
    else:
        result = "Deforestation 🌳➡️❌"

    recommended_trees = None
    if change < 0:
        recommended_trees = recommend_trees(percent_after)

    mask_before_path = f"{output_dir}/mask_before.png"
    mask_after_path = f"{output_dir}/mask_after.png"

    cv2.imwrite(mask_before_path, mask_before)
    cv2.imwrite(mask_after_path, mask_after)

    geo_map_path = f"{output_dir}/geospatial_map.png"
    generate_geospatial_map(mask_before, mask_after, geo_map_path)

    return {
        "before_percent": round(percent_before, 2),
        "after_percent": round(percent_after, 2),
        "change": round(change, 2),
        "result": result,
        "mask_before": mask_before_path,
        "mask_after": mask_after_path,
        "geo_map": geo_map_path,
        "trees": recommended_trees
    }



# ---------- NEW FUNCTION (ADD THIS BELOW) ----------
def analyze_single_image(image_path, output_dir="static/outputs"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    forest_pixels = np.sum(mask > 0)
    total_pixels = mask.size

    forest_percent = (forest_pixels / total_pixels) * 100

    if forest_percent >= 5:
        result = "Forested Area 🌳"
    else:
        result = "Deforested / Non-Forest Area 🏜️"

    # 🟢 VISUAL COLOR MASK (FIX)
    height, width = mask.shape
    color_mask = np.ones((height, width, 3), dtype=np.uint8) * 180
    color_mask[mask > 0] = [0, 255, 0]

    mask_path = f"{output_dir}/single_forest_mask.png"
    cv2.imwrite(mask_path, color_mask)

    return {
        "forest_percent": round(forest_percent, 2),
        "result": result,
        "mask": mask_path
    }

def analyze_single_image_ml(image_path, output_dir="static/outputs"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img_norm = img / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    pred = ml_model.predict(img_input)[0]

    # Binary mask at ORIGINAL model resolution (256x256)
    binary_mask = (pred > 0.7).astype(np.uint8)

# ✅ CORRECT forest percentage calculation
    forest_percent = (np.sum(binary_mask) / binary_mask.size) * 100

# Resize ONLY for visualization
    mask_vis = cv2.resize(binary_mask * 255, (512, 512))

# Clean color mask (BLACK background)
    color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    color_mask[mask_vis > 0] = [0, 255, 0]


    mask_path = f"{output_dir}/ml_forest_mask.png"
    
    cv2.imwrite(mask_path, color_mask)

    if forest_percent > 40:
        result = "Forest-Dominant Region 🌳"
    elif forest_percent > 15:
        result = "Mixed Land Cover 🌾"
    else:
        result = "Low Forest / Non-Forest Area 🏜️"

    recommended_trees = recommend_trees(forest_percent)


   
    return {
    "forest_percent": round(forest_percent, 2),
    "result": result,
    "mask": mask_path,
    "trees": recommended_trees
}


