# from .constants import DETECTION_TIME

# from collections import defaultdict
# from ultralytics import YOLO
# from easyocr import Reader
# import logging
# import time
# import cv2
# import re

# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)
# reader = Reader(['en'])

# video_path = "./videos/full-masuk-keluar.mp4"
# # video_path = "./videos/keluar.mp4"

# class PlateRecognizer:

#     def __init__(self, model_path: str = "./model/best.pt"):
#         self.model = YOLO(model_path)
#         self.cap = cv2.VideoCapture(video_path)

#         # Get the video properties
#         frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Define the desired width for resizing
#         desired_width = 720

#         # Calculate the scaling factor and new dimensions
#         scale = desired_width / frame_width
#         self.new_width = int(frame_width * scale)
#         self.new_height = int(frame_height * scale)

#         # Dictionary to store detection times for each bbox
#         self.detection_times = defaultdict(float)

#     def run(self):
#         camera_in = self.camera()
#         camera_out = self.camera()

#     def camera(self):

#         log.info("Starting video processing...")
#         while self.cap.isOpened():
#             # Read a frame from the video
#             success, frame = self.cap.read()

#             if success:
#                 # Resize the frame
#                 resized_frame = cv2.resize(frame, (self.new_width, self.new_height))

#                 # Run YOLOv8 inference on the resized frame
#                 results = self.model(resized_frame, verbose=False)

#                 current_time = time.time()

#                 for r in results:
#                     boxes = r.boxes
#                     for box in boxes:
#                         b = box.xyxy[0].tolist()

#                         text = self.extract_text(resized_frame, b)

#                         log.info(f"Detected text: {text}")

#                 # Visualize the results on the resized frame
#                 annotated_frame = results[0].plot()

#                 # Ensure the frame is in the correct format for OpenCV
#                 if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
#                     annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

#                 # Display the annotated frame
#                 try:
#                     cv2.imshow("YOLOv8 Inference", annotated_frame)
#                 except cv2.error as e:
#                     log.error(f"Error displaying frame: {e}")
#                     log.error("Continuing processing without display...")

#                 # Break the loop if 'q' is pressed
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     log.info("Video processing stopped by user.")
#                     break
#             else:
#                 # Break the loop if the end of the video is reached
#                 break

#         # Release the video capture object and close the display window
#         self.cap.release()
#         cv2.destroyAllWindows()

#         log.info("Video processing complete.")

#     def extract_text(self, frame, bbox):

#         x1, y1, x2, y2 = map(int, bbox)
#         roi = frame[y1:y2, x1:x2]
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

#         # Perform OCR on the entire image
#         results = reader.readtext(gray)

#         # Extract text from results
#         text = []
#         for (bbox, content, prob) in results:
#             text.append(content)

#         # Join all detected text into a single string
#         full_text = ' '.join(text)

#         # Process the text to match the desired format
#         processed_plates = self.process_plate_text(full_text)

#         print("=============")
#         # print("full_text:", full_text)
#         print("processed_plates:", processed_plates)


#         # return processed_plates
#         return full_text

#     def process_plate_text(self, text):

#         # Split the text into lines
#         lines = text.split('\n')
#         processed_plates = []
#         for line in lines:
#             # Remove any non-alphanumeric characters
#             clean_line = re.sub(r'[^a-zA-Z0-9]', ' ', line)
#             # Split the line into parts
#             parts = clean_line.split()

#             if len(parts) >= 3:
#                 # Process each part according to the rules
#                 part1 = ''.join(filter(str.isalpha, parts[0]))[
#                     :2].upper()  # First part: up to 2 letters
#                 # Middle part: up to 4 digits
#                 part2 = ''.join(filter(str.isdigit, ''.join(parts[1:-1])))[:4]
#                 # Last part: up to 2 letters
#                 part3 = ''.join(filter(str.isalpha, parts[-1]))[:2].upper()

#                 # Ensure part3 is not empty, if it is, try to find letters in previous parts
#                 if not part3:
#                     for part in parts[2:-1]:
#                         part3 = ''.join(filter(str.isalpha, part))[:2].upper()
#                         if part3:
#                             break

#                 # Only add the processed plate if all parts are valid
#                 if part1 and part2 and part3:
#                     processed_plate = f"{part1} {part2} {part3}"
#                     processed_plates.append(processed_plate)

#         return '\n'.join(processed_plates)
import logging as log
from .constants import DETECTION_TIME
from collections import defaultdict
from ultralytics import YOLO
from easyocr import Reader
import logging
import time
import cv2
import re
import threading

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
reader = Reader(['en'])


class PlateRecognizer:

    def __init__(self,
                 model_path_1: str = "./model/best-1.pt",
                 model_path_2: str = "./model/best-2.pt",
                 video_path_in: str = "./videos/masuk.mp4",
                 video_path_out: str = "./videos/keluar.mp4"):
        try:
            self.model_1 = YOLO(model_path_1)
            self.model_2 = YOLO(model_path_2)
        except Exception as e:
            log.error(f"Gagal memuat model YOLO: {e}")
            raise e

        self.cap_in = cv2.VideoCapture(video_path_in)
        if not self.cap_in.isOpened():
            log.error(f"Gagal membuka video {video_path_in}")
            raise IOError(f"Gagal membuka video {video_path_in}")

        self.cap_out = cv2.VideoCapture(video_path_out)
        if not self.cap_out.isOpened():
            log.error(f"Gagal membuka video {video_path_out}")
            raise IOError(f"Gagal membuka video {video_path_out}")

        frame_width = int(self.cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

        desired_width = 720
        scale = desired_width / frame_width
        self.new_width = int(frame_width * scale)
        self.new_height = int(frame_height * scale)

        self.detection_times_in = defaultdict(float)
        self.detection_times_out = defaultdict(float)
        self.data_plates = {}
        self.lock = threading.Lock()
        self.state_plates = {}

    def run(self):
        thread_in = threading.Thread(target=self.camera_in)
        thread_out = threading.Thread(target=self.camera_out)
        thread_in.start()
        thread_out.start()
        thread_in.join()
        thread_out.join()

    def camera_in(self):
        self._process_camera(self.cap_in, "Camera in", self.model_1)

    def camera_out(self):
        self._process_camera(self.cap_out, "Camera out", self.model_2)

    def _process_camera(self, cap, camera_name, model):
        log.info(f"Memulai pemrosesan video {camera_name}...")
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                resized_frame = cv2.resize(
                    frame, (self.new_width, self.new_height))
                try:
                    results = model(resized_frame, verbose=False)
                except AttributeError as e:
                    log.error(f"Error saat memproses frame dengan model: {e}")
                    break
                except Exception as e:
                    log.error(f"Error tak terduga saat memproses frame: {e}")
                    continue

                current_time = time.time()

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0].tolist()
                        text = self.extract_text(resized_frame, b)
                        log.info(f"Detected text: {text}")

                        if camera_name == 'Camera in':
                            self.handle_camera_in(text, current_time)
                        else:
                            self.handle_camera_out(text, current_time)

                try:
                    annotated_frame = results[0].plot()
                    if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                        annotated_frame = cv2.cvtColor(
                            annotated_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(
                        f"YOLOv8 Inference {camera_name}", annotated_frame)
                except Exception as e:
                    log.error(f"Error saat menampilkan frame: {e}")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Video processing stopped by user.")
                    print("DATA PLATES:", self.data_plates)
                    print("STATE PLATES:", self.state_plates)
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        log.info(f"Pemrosesan video {camera_name} selesai.")

    def handle_camera_in(self, text, current_time):
        with self.lock:
            if text not in self.data_plates and text != "":
                self.data_plates[text] = current_time
                self.state_plates[text] = True
                log.info(f"Plate '{text}' ditambahkan ke data_plates.")

    def handle_camera_out(self, text, current_time):
        with self.lock:
            if text in self.data_plates:
                time_in = self.data_plates[text]
                time_out = current_time
                log.info(
                    f"Plate '{text}' masuk pada {time.ctime(time_in)} dan keluar pada {time.ctime(time_out)}.")
                del self.data_plates[text]
                del self.state_plates[text]

    def extract_text(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        results = reader.readtext(gray)
        text = [content for (_, content, _) in results]
        full_text = ' '.join(text)
        processed_plates = self.process_plate_text(full_text)

        return processed_plates

    def process_plate_text(self, text):
        lines = text.split('\n')
        processed_plates = []
        for line in lines:
            clean_line = re.sub(r'[^a-zA-Z0-9]', ' ', line)
            parts = clean_line.split()

            if len(parts) >= 3:
                part1 = ''.join(filter(str.isalpha, parts[0]))[:2].upper()
                part2 = ''.join(filter(str.isdigit, ''.join(parts[1:-1])))[:4]
                part3 = ''.join(filter(str.isalpha, parts[-1]))[:2].upper()

                if not part3:
                    for part in parts[2:-1]:
                        part3 = ''.join(filter(str.isalpha, part))[:2].upper()
                        if part3:
                            break

                if part1 and part2 and part3:
                    processed_plate = f"{part1} {part2} {part3}"
                    processed_plates.append(processed_plate)

        return '\n'.join(processed_plates)
