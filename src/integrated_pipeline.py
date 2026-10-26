
import os
import sys
import json
import csv
from pathlib import Path
import importlib.util
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
import textwrap
# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\admin\Desktop\CCTV_Integrated_Pipeline"
)

CLOTHING_MODEL_PATH = Path(
    r"C:\Users\admin\Desktop\Clothing_Pipeline\training_results_v6\best_clothing_model_v2.keras"
)

CLOTHING_CLASSES_PATH = Path(
    r"C:\Users\admin\Desktop\Clothing_Pipeline\training_results_v6\class_names.json"
)

PATTERN_MODEL_PATH = Path(
    r"C:\Users\admin\Desktop\布料偵測資料GitHub\clothing-detection-0.2\runs\classify\runs\classify\pattern_baseline-3\weights\best.pt"
)

COLOR_SCRIPT_PATH = Path(
    r"C:\Users\admin\Desktop\Color_Pipeline\Final_Pipeline.py"
)

INPUT_DIR = Path(
    r"C:\Users\admin\Desktop\CCTV_Integrated_Pipeline\Input (Filtered_Torso_Crops)"
)

OUTPUT_DIR = BASE_DIR / "integrated_results"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"
RESULTS_CSV = OUTPUT_DIR / "integrated_predictions.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"


CLOTHING_IMAGE_SIZE = 224
PATTERN_IMAGE_SIZE = 224

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# 2. VALIDATE FILES
# ============================================================

for path in [
    CLOTHING_MODEL_PATH,
    CLOTHING_CLASSES_PATH,
    PATTERN_MODEL_PATH,
    COLOR_SCRIPT_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n\n"
            "Check the path before running the integration."
        )

if not INPUT_DIR.is_dir():
    raise FileNotFoundError(
        f"Input directory not found:\n{INPUT_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Make the existing color script's relative paths predictable.
os.chdir(BASE_DIR)

print("=" * 70)
print("INTEGRATED CCTV CLOTHING ANALYSIS")
print("=" * 70)

# ============================================================
# 3. LOAD CLOTHING MODEL
# ============================================================

print("\nLoading V6 clothing classifier...")

with open(CLOTHING_CLASSES_PATH, "r", encoding="utf-8") as f:
    clothing_classes = json.load(f)

clothing_model = tf.keras.models.load_model(
    CLOTHING_MODEL_PATH,
    compile=False,
)

if tuple(clothing_model.input_shape[1:3]) != (
    CLOTHING_IMAGE_SIZE,
    CLOTHING_IMAGE_SIZE,
):
    raise ValueError(
        "Clothing model input dimensions do not match "
        f"{CLOTHING_IMAGE_SIZE}x{CLOTHING_IMAGE_SIZE}: "
        f"{clothing_model.input_shape}"
    )

print("Clothing classes:", clothing_classes)

# ============================================================
# 4. LOAD PATTERN MODEL
# ============================================================

print("\nLoading pattern classifier...")

pattern_model = YOLO(str(PATTERN_MODEL_PATH))

# Select the device for the YOLO pattern classifier


device = 0 if torch.cuda.is_available() else "cpu"

print(f"Pattern classifier device: {device}")

print("Pattern model loaded.")
print("Pattern inference device:", device)

# ============================================================
# 5. LOAD EXISTING COLOR PIPELINE
# ============================================================

print("\nLoading existing SegFormer color pipeline...")

# Importing this module loads its existing segmentation model.
# Its __main__ loop will NOT run when imported.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load the color pipeline from its external file path
spec = importlib.util.spec_from_file_location(
    "color_pipeline",
    str(COLOR_SCRIPT_PATH),
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Cannot load color pipeline: {COLOR_SCRIPT_PATH}"
    )

color_pipeline = importlib.util.module_from_spec(spec)
sys.modules["color_pipeline"] = color_pipeline
spec.loader.exec_module(color_pipeline)

# Redirect the existing color pipeline's visualizations.
color_pipeline.VIS_OUTPUT_FOLDER = str(
    OUTPUT_DIR / "color_segmentations"
)
color_pipeline.OUTPUT_CSV = str(
    OUTPUT_DIR / "color_pipeline_predictions.csv"
)

Path(color_pipeline.VIS_OUTPUT_FOLDER).mkdir(
    parents=True,
    exist_ok=True,
)

print("Color segmentation model loaded.")

# ============================================================
# 6. HELPERS
# ============================================================

def torch_cuda_available():
    
    return torch.cuda.is_available()


def load_font(size=20):
    for font_path in [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
    ]:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                pass

    return ImageFont.load_default()


def predict_clothing(image_path):
    """Predict one of the existing V6 clothing classes."""

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(
            (CLOTHING_IMAGE_SIZE, CLOTHING_IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )
        image_array = np.asarray(img, dtype=np.float32)

    # EfficientNet preprocessing is already included in the model.
    # Do not divide pixel values by 255.
    probabilities = clothing_model.predict(
        np.expand_dims(image_array, axis=0),
        verbose=0,
    )[0]

    index = int(np.argmax(probabilities))

    return (
        clothing_classes[index],
        float(probabilities[index]),
    )


def predict_pattern(image_path):
    """Predict one pattern class using the existing YOLO model."""

    result = pattern_model.predict(
        source=str(image_path),
        imgsz=PATTERN_IMAGE_SIZE,
        device=device,
        verbose=False,
    )[0]

    if result.probs is None:
        raise RuntimeError("Pattern model returned no probabilities.")

    index = int(result.probs.top1)

    return (
        result.names[index],
        float(result.probs.top1conf),
    )


def predict_colors(image_path):
    """Call the existing SegFormer + K-means + HSV pipeline."""

    # This function saves the existing side-by-side segmentation
    # visualization and returns its color prediction string.
    return color_pipeline.process_and_visualize(
        str(image_path),
        image_path.name,
    )



def save_combined_annotation(
    image_path,
    output_path,
    clothing,
    clothing_confidence,
    colors,
    pattern,
    pattern_confidence,
):
    """Save predictions in a professional panel above the torso crop.
    Long prediction text wraps automatically and panel height adapts.
    """

    
    
    

    # Open original image
    with Image.open(image_path) as source:
        img = source.convert("RGB")

    # Resize proportionally if needed
    max_width = 900
    if img.width > max_width:
        new_height = round(img.height * max_width / img.width)
        img = img.resize(
            (max_width, new_height),
            Image.Resampling.LANCZOS,
        )

    width = img.width

    # Professional styling
    background = (255, 255, 255)
    text_color = (40, 45, 52)
    accent_color = (42, 91, 145)
    divider_color = (220, 224, 230)

    # Load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
        label_font = ImageFont.truetype("arial.ttf", 15)
        value_font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        value_font = ImageFont.load_default()

    draw_helper = ImageDraw.Draw(
        Image.new("RGB", (width, 10), background)
    )

    left_margin = 18
    right_margin = 18
    label_gap = 8
    top_margin = 14
    title_gap = 12
    row_gap = 9
    line_gap = 5
    bottom_margin = 14

    # Calculate label width and available value width
    labels = ["Clothing:", "Colors:", "Pattern:"]
    label_widths = [
        draw_helper.textbbox((0, 0), label, font=label_font)[2]
        for label in labels
    ]
    max_label_width = max(label_widths)

    value_x = left_margin + max_label_width + label_gap
    value_width = max(80, width - value_x - right_margin)

    def wrap_to_pixel_width(text, font, max_width):
        """Wrap text based on actual rendered pixel width."""
        words = str(text).split()
        if not words:
            return [""]

        lines = []
        current_line = ""

        for word in words:
            candidate = (
                word if not current_line
                else current_line + " " + word
            )

            candidate_width = draw_helper.textbbox(
                (0, 0), candidate, font=font
            )[2]

            if candidate_width <= max_width:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Handle a single unusually long word
                    fragment = ""
                    for character in word:
                        test = fragment + character
                        test_width = draw_helper.textbbox(
                            (0, 0), test, font=font
                        )[2]

                        if test_width <= max_width or not fragment:
                            fragment = test
                        else:
                            lines.append(fragment)
                            fragment = character

                    current_line = fragment

        if current_line:
            lines.append(current_line)

        return lines or [""]

    # Prepare prediction rows
    raw_rows = [
        ("Clothing:", f"{clothing} ({clothing_confidence:.1%})"),
        ("Colors:", str(colors)),
        ("Pattern:", f"{pattern} ({pattern_confidence:.1%})"),
    ]

    rows = []
    for label, value in raw_rows:
        wrapped_lines = wrap_to_pixel_width(
            value, value_font, value_width
        )
        rows.append((label, wrapped_lines))

    # Calculate panel height from actual number of lines
    title_bbox = draw_helper.textbbox(
        (0, 0), "CLOTHING ANALYSIS", font=title_font
    )
    title_height = title_bbox[3] - title_bbox[1]

    value_bbox = draw_helper.textbbox(
        (0, 0), "Ag", font=value_font
    )
    line_height = value_bbox[3] - value_bbox[1] + line_gap

    panel_height = top_margin + title_height + title_gap

    for _, lines in rows:
        panel_height += len(lines) * line_height + row_gap

    panel_height += bottom_margin

    # Draw panel
    panel = Image.new(
        "RGB", (width, panel_height), background
    )
    draw = ImageDraw.Draw(panel)

    draw.rectangle(
        (0, 0, width, 5),
        fill=accent_color,
    )

    draw.text(
        (left_margin, top_margin),
        "CLOTHING ANALYSIS",
        font=title_font,
        fill=accent_color,
    )

    y = top_margin + title_height + title_gap

    for label, lines in rows:
        draw.text(
            (left_margin, y),
            label,
            font=label_font,
            fill=text_color,
        )

        for line_index, line in enumerate(lines):
            line_y = y + line_index * line_height

            draw.text(
                (value_x, line_y),
                line,
                font=value_font,
                fill=text_color,
            )

        y += len(lines) * line_height + row_gap

    # Combine panel, divider, and original crop
    result = Image.new(
        "RGB",
        (width, panel_height + 2 + img.height),
        background,
    )

    result.paste(panel, (0, 0))

    divider_draw = ImageDraw.Draw(result)
    divider_draw.rectangle(
        (0, panel_height, width, panel_height + 1),
        fill=divider_color,
    )

    result.paste(img, (0, panel_height + 2))

    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=95)



# ============================================================
# 7. PROCESS INPUT IMAGES
# ============================================================

image_paths = sorted(
    [
        p for p in INPUT_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ],
    key=lambda p: str(p).lower(),
)

if not image_paths:
    raise RuntimeError(f"No images found in {INPUT_DIR}")

print(f"\nFound {len(image_paths)} images.")
print("\nStarting integrated inference...\n")

records = []

for i, image_path in enumerate(image_paths, start=1):
    print(f"[{i}/{len(image_paths)}] {image_path.name}")

    record = {
        "filename": str(image_path.relative_to(INPUT_DIR)),
        "clothing": None,
        "clothing_confidence": None,
        "colors": None,
        "pattern": None,
        "pattern_confidence": None,
        "errors": "",
    }

    errors = []

    # --------------------------------------------------------
    # CLOTHING
    # --------------------------------------------------------

    try:
        clothing, clothing_confidence = predict_clothing(image_path)

        record["clothing"] = clothing
        record["clothing_confidence"] = clothing_confidence

        print(
            f"  Clothing: {clothing} "
            f"({clothing_confidence:.1%})"
        )

    except Exception as exc:
        errors.append(f"clothing: {exc}")
        print(f"  Clothing ERROR: {exc}")

    # --------------------------------------------------------
    # COLORS
    # --------------------------------------------------------

    try:
        colors = predict_colors(image_path)

        record["colors"] = colors

        print(f"  Colors: {colors}")

    except Exception as exc:
        errors.append(f"colors: {exc}")
        print(f"  Colors ERROR: {exc}")

    # --------------------------------------------------------
    # PATTERN
    # --------------------------------------------------------

    try:
        pattern, pattern_confidence = predict_pattern(image_path)

        record["pattern"] = pattern
        record["pattern_confidence"] = pattern_confidence

        print(
            f"  Pattern: {pattern} "
            f"({pattern_confidence:.1%})"
        )

    except Exception as exc:
        errors.append(f"pattern: {exc}")
        print(f"  Pattern ERROR: {exc}")

    # --------------------------------------------------------
    # COMBINED ANNOTATION
    # --------------------------------------------------------

    try:
        combined_path = (
            ANNOTATED_DIR
            / f"{image_path.stem}_integrated.jpg"
        )

        save_combined_annotation(
            image_path=image_path,
            output_path=combined_path,
            clothing=record["clothing"] or "ERROR",
            clothing_confidence=(
                record["clothing_confidence"] or 0.0
            ),
            colors=record["colors"] or "ERROR",
            pattern=record["pattern"] or "ERROR",
            pattern_confidence=(
                record["pattern_confidence"] or 0.0
            ),
        )

        record["combined_image"] = str(combined_path)

    except Exception as exc:
        errors.append(f"annotation: {exc}")

    record["errors"] = " | ".join(errors)
    records.append(record)

# ============================================================
# 8. SAVE UNIFIED RESULTS
# ============================================================

results_df = pd.DataFrame(records)

results_df.to_csv(
    RESULTS_CSV,
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "total_images": len(image_paths),
    "clothing_successes": int(results_df["clothing"].notna().sum()),
    "color_successes": int(results_df["colors"].notna().sum()),
    "pattern_successes": int(results_df["pattern"].notna().sum()),
    "images_with_errors": int(
        (results_df["errors"].fillna("") != "").sum()
    ),
    "clothing_prediction_counts": (
        results_df["clothing"].value_counts(dropna=True).to_dict()
    ),
    "pattern_prediction_counts": (
        results_df["pattern"].value_counts(dropna=True).to_dict()
    ),
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print("INTEGRATED PIPELINE COMPLETED")
print("=" * 70)
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nCSV:", RESULTS_CSV)
print("Summary:", SUMMARY_JSON)
print("Combined annotated images:", ANNOTATED_DIR)
print(
    "Color segmentation visualizations:",
    color_pipeline.VIS_OUTPUT_FOLDER,
)
