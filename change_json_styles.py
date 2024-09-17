import json
import shutil
import signal
import sys
import time
import os
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class LandmarksHandler(FileSystemEventHandler):
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.json'):
            # 防止出现文件读写冲突或未完全写入
            time.sleep(1)
            self.process_landmarks(event.src_path)

    def process_landmarks(self, file_path):
        try:
            # Read landmarks file
            with open(file_path, 'r') as file:
                landmarks_data = json.load(file)

            # Indices to extract from landmarks
            indices = [0, 2, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

            # Corresponding keys in goal dictionary
            keys = [
                "EYE_DEF", "eye_L_old", "eye_R_old", "eye_base_old",
                "Character1_RightShoulder", "Character1_LeftShoulder", "Character1_LeftArm", "Character1_RightArm",
                "Character1_LeftForeArm", "Character1_RightForeArm", "Character1_LeftHandPinky4", "Character1_RightHandPinky4",
                "Character1_LeftHandMiddle4", "Character1_RightHandMiddle4", "Character1_LiftHandThumb4", "Character1_RightHandThumb4",
                "Character1_Hips", "Character1_LeftUpLeg", "Character1_RightUpLeg", "Character1_LeftLeg", 
                "Character1_RightLeg", "Character1_LeftFoot", "Character1_RightFoot", "Character1_LeftToeBase", 
                "Character1_RightToeBase"
            ]

            # Initialize an empty posedlist
            posedlist = []

            # Extract landmarks and create posedlist entries
            for i, landmarks_entry in enumerate(landmarks_data):
                dic = {}
                landmarks = landmarks_entry['landmarks']
                
                for j, key in enumerate(keys):
                    idx = indices[j]
                    dic[key] = {
                        'x': landmarks[idx]['x'],
                        'y': landmarks[idx]['y'],
                        'z': landmarks[idx]['z']
                    }
                
                posedlist.append({"id": str(i), "dic": dic})

            # Create the final goal structure
            goal_data = {"posedlist": posedlist}
            goal_data["protoName"] = "JsonSend"

            # Output file path
            output_file_path = os.path.join(self.output_folder, os.path.basename(file_path))

            # Save updated goal file
            with open(output_file_path, 'w') as file:
                json.dump(goal_data, file, indent=4)

            print(f"Processed {file_path} and saved to {output_file_path}")

        except PermissionError as e:
            print(f"PermissionError: {e}. Retrying in 1 second...")
            time.sleep(1)
            self.process_landmarks(file_path)

def monitor_folder(input_folder, output_folder):
    event_handler = LandmarksHandler(output_folder)
    observer = Observer()
    observer.schedule(event_handler, path=input_folder, recursive=False)
    observer.start()
    print(f"Monitoring {input_folder} for new JSON files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def clean_folders(folders, max_files=720, delete_count=700):
    for folder in folders:
        files = sorted(os.listdir(folder), key=lambda x: os.path.getmtime(os.path.join(folder, x)))
        if len(files) > max_files:
            files_to_delete = files[:delete_count]
            for filename in files_to_delete:
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

def clean_all_folders(folders):
    for folder in folders:
        files = sorted(os.listdir(folder), key=lambda x: os.path.getmtime(os.path.join(folder, x)))
        max_files = len(files)
        files_to_delete = files[:max_files]
        for filename in files_to_delete:
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


if __name__ == "__main__":
    input_folder = './json_save'  # Replace with your input folder path
    output_folder = './json_output'  # Replace with your output folder path
    folders_to_clean = [input_folder, output_folder]

    # Start monitoring folder in a separate thread
    monitor_thread = Thread(target=monitor_folder, args=(input_folder, output_folder))
    monitor_thread.start()

    # Clean folders every 5 seconds
    try:
        while True:
            time.sleep(300)
            clean_folders(folders_to_clean)
            print("Cleaned folders.")
    except KeyboardInterrupt:
        print("Stopping...")
        monitor_thread.join()