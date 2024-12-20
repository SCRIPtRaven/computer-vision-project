import os

import cv2


def split_image(image, window_size, step_size):
    h, w, _ = image.shape
    patches = []

    for y in range(0, h - window_size + 1, step_size):
        for x in range(0, w - window_size + 1, step_size):
            patch = image[y:y + window_size, x:x + window_size]
            patches.append(((x, y), patch))

    return patches


def process_directory(data_dir, output_dir, window_size, step_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ["train", "test"]:
        for folder in ["original", "ground_truth"]:
            split_dir = os.path.join(data_dir, folder, split)
            output_split_dir = os.path.join(output_dir, folder, split)

            for root, _, files in os.walk(split_dir):
                relative_path = os.path.relpath(root, split_dir)
                save_dir = os.path.join(output_split_dir, relative_path)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                for file in files:
                    if file.endswith(('.jpg', '.tif')):
                        file_path = os.path.join(root, file)
                        image = cv2.imread(file_path)

                        if image is None:
                            print(f"Could not read {file_path}, skipping...")
                            continue

                        patches = split_image(image, window_size, step_size)

                        for (x, y), patch in patches:
                            patch_filename = f"{os.path.splitext(file)[0]}_x{x}_y{y}.png"
                            patch_path = os.path.join(save_dir, patch_filename)
                            cv2.imwrite(patch_path, patch)


data_dir = "data"
output_dir = "processed_data"
window_size = 1024
step_size = 512


def main():
    process_directory(data_dir, output_dir, window_size, step_size)
    print("Processing completed")


if __name__ == "__main__":
    main()
