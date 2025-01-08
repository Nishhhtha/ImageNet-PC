import os
import cv2
import numpy as np
import albumentations as A
from concurrent.futures import ThreadPoolExecutor

# Helper function: Read bounding boxes from a label file
def read_bboxes(label_file):
    bboxes = []
    with open(label_file, 'r') as file:
        for line in file:
            _, x_center, y_center, width, height = map(float, line.strip().split())
            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)
            bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

# Camouflage
def apply_camouflaging(image, bbox, severity):
    try:
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, image.shape[1]), min(y_max, image.shape[0])
        roi = image[y_min:y_max, x_min:x_max]
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 255
        inpainted_image = cv2.inpaint(image, mask[:, :, 0], inpaintRadius=severity * 10, flags=cv2.INPAINT_TELEA)
        blend_ratios = {1: 0.55, 2: 0.67, 3: 0.74, 4: 0.82, 5: 0.90}
        blended_roi = cv2.addWeighted(roi, 1 - blend_ratios[severity], inpainted_image[y_min:y_max, x_min:x_max], blend_ratios[severity], 0)
        image[y_min:y_max, x_min:x_max] = blended_roi
    except Exception as e:
        print(f"Error applying camouflaging: {str(e)}")
    return image

# Fingerprint Noise
def apply_fingerprint_noise(image, bbox, severity, fingerprint_texture):
    try:
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        fingerprint_resized = cv2.resize(fingerprint_texture, (cropped.shape[1], cropped.shape[0]))
        alpha = min(severity * 0.15, 1.0)
        blended = cv2.addWeighted(cropped, 1 - alpha, fingerprint_resized, alpha, 0)
        image[y_min:y_max, x_min:x_max] = blended
    except Exception as e:
        print(f"Error applying fingerprint noise: {str(e)}")
    return image

# Illumination Variation
def apply_illumination_variation(image, bbox, severity):
    try:
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(0.03 * severity, 0.085 * severity),
                contrast_limit=(0.03 * severity, .085 * severity),
                p=1.0
            )
        ])
        augmented = transform(image=cropped)
        image[y_min:y_max, x_min:x_max] = augmented['image']
    except Exception as e:
        print(f"Error applying illumination variation: {str(e)}")
    return image

# Dust and Scratches
def apply_dust_scratches(image, bbox, severity):
    try:
        x_min, y_min, x_max, y_max = bbox
        corrupted_image = image.copy()
        num_scratches = severity * 3
        num_dust_particles = severity * 20
        for _ in range(num_scratches):
            x_start = np.random.randint(x_min, x_max)
            y_start = np.random.randint(y_min, y_max)
            length = np.random.randint(10, 20 * severity)
            angle = np.random.uniform(0, 2 * np.pi)
            x_end = int(x_start + length * np.cos(angle))
            y_end = int(y_start + length * np.sin(angle))
            scratch_color = (np.random.randint(200, 256), np.random.randint(200, 256), np.random.randint(200, 256))
            cv2.line(corrupted_image, (x_start, y_start), (x_end, y_end), scratch_color, 1)
        for _ in range(num_dust_particles):
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            dust_color = (np.random.randint(200, 256), np.random.randint(200, 256), np.random.randint(200, 256))
            cv2.circle(corrupted_image, (x, y), 1, dust_color, -1)
        image[y_min:y_max, x_min:x_max] = corrupted_image[y_min:y_max, x_min:x_max]
    except Exception as e:
        print(f"Error applying dust and scratches: {str(e)}")
    return image

# Lens Flare
def apply_lens_flare(image, bbox, severity):
    try:
        center_x = np.random.randint(bbox[0], bbox[2])
        center_y = np.random.randint(bbox[1], bbox[3])
        radius = severity * 20
        color = (np.random.randint(200, 256), np.random.randint(200, 256), np.random.randint(200, 256))
        cv2.circle(image, (center_x, center_y), radius, color, -1)
    except Exception as e:
        print(f"Error applying lens flare: {str(e)}")
    return image

# Occlusion
def apply_partial_occlusion(image, bbox, severity):
    try:
        x_min, y_min, x_max, y_max = bbox
        for _ in range(severity):
            occlusion_x = np.random.randint(x_min, x_max)
            occlusion_y = np.random.randint(y_min, y_max)
            patch_width = int(0.2 * (x_max - x_min) * severity / 5)
            patch_height = int(0.2 * (y_max - y_min) * severity / 5)
            color = tuple(np.random.randint(0, 256, size=3))
            image[occlusion_y:occlusion_y+patch_height, occlusion_x:occlusion_x+patch_width] = color
    except Exception as e:
        print(f"Error applying partial occlusion: {str(e)}")
    return image

# Object Focus Shift
def apply_object_focus_shift(image, bbox, severity):
    try:
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        blurred = cv2.GaussianBlur(cropped, (severity * 2 + 1, severity * 2 + 1), severity)
        image[y_min:y_max, x_min:x_max] = blurred
    except Exception as e:
        print(f"Error applying object focus shift: {str(e)}")
    return image

# Process Image
def process_image(image_name, class_dir, dataset_dir, labels_dir, corruption_dirs, fingerprint_texture, severity=1):
    image_path = os.path.join(dataset_dir, class_dir, image_name)
    label_file = os.path.join(labels_dir, class_dir, image_name.replace('.jpg', '.txt'))
    image = cv2.imread(image_path)
    global image_height, image_width
    image_height, image_width = image.shape[:2]
    bboxes = read_bboxes(label_file)

    for corruption, output_dir in corruption_dirs.items():
        corrupted_image = image.copy()
        for bbox in bboxes:
            if corruption == 'camouflage':
                corrupted_image = apply_camouflaging(corrupted_image, bbox, severity)
            elif corruption == 'fingerprint':
                corrupted_image = apply_fingerprint_noise(corrupted_image, bbox, severity, fingerprint_texture)
            elif corruption == 'illumination_variation':
                corrupted_image = apply_illumination_variation(corrupted_image, bbox, severity)
            elif corruption == 'dust_scratches':
                corrupted_image = apply_dust_scratches(corrupted_image, bbox, severity)
            elif corruption == 'lens_flare':
                corrupted_image = apply_lens_flare(corrupted_image, bbox, severity)
            elif corruption == 'occlusion':
                corrupted_image = apply_partial_occlusion(corrupted_image, bbox, severity)
            elif corruption == 'focus_shift':
                corrupted_image = apply_object_focus_shift(corrupted_image, bbox, severity)
        output_path = os.path.join(output_dir, image_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, corrupted_image)

# Main Execution Block
if __name__ == "__main__":
    dataset_dir = './dataset'
    labels_dir = './labels'
    corruption_dirs = {
        'camouflage': './corrupted/camouflage',
        'fingerprint': './corrupted/fingerprint',
        'illumination_variation': './corrupted/illumination_variation',
        'dust_scratches': './corrupted/dust_scratches',
        'lens_flare': './corrupted/lens_flare',
        'occlusion': './corrupted/occlusion',
        'focus_shift': './corrupted/focus_shift'
    }
    fingerprint_texture = cv2.imread('./fingerprint.jpg')

    with ThreadPoolExecutor(max_workers=10) as executor:
        for class_dir in os.listdir(dataset_dir):
            for image_name in os.listdir(os.path.join(dataset_dir, class_dir)):
                executor.submit(process_image, image_name, class_dir, dataset_dir, labels_dir, corruption_dirs, fingerprint_texture)
