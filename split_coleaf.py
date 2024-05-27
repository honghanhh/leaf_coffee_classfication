import os
import shutil
from sklearn.model_selection import train_test_split

def create_dirs(base_dir, dirs):
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

def split_dataset(input_dir, output_dir):
    # Directories for the output datasets
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    # Create directories for train, val, and test sets
    create_dirs(output_dir, ['train', 'val', 'test'])

    # Loop through each class in the input directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if os.path.isdir(class_dir):
            # List all files in the class directory
            files = os.listdir(class_dir)
            files = [os.path.join(class_dir, f) for f in files if os.path.isfile(os.path.join(class_dir, f))]

            # Split into train+val (80%) and test (20%)
            train_val_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

            # Further split train_val into train (90%) and val (10%)
            train_files, val_files = train_test_split(train_val_files, test_size=0.1, random_state=42)

            # Create class subdirectories in train, val, and test directories
            create_dirs(train_dir, [class_name])
            create_dirs(val_dir, [class_name])
            create_dirs(test_dir, [class_name])

            # Copy files to the respective directories
            for f in train_files:
                shutil.copy(f, os.path.join(train_dir, class_name))

            for f in val_files:
                shutil.copy(f, os.path.join(val_dir, class_name))

            for f in test_files:
                shutil.copy(f, os.path.join(test_dir, class_name))

if __name__ == '__main__':
    input_directory = 'data/Co-Leaf'
    output_directory = 'data/CoLeaf'
    split_dataset(input_directory, output_directory)