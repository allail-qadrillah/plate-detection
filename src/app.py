# from .servo import Servo

# import logging as log
# from collections import defaultdict
# from ultralytics import YOLO
# from easyocr import Reader
# import logging
# import time
# import cv2
# import re
# import threading

# # logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.ERROR)
# log = logging.getLogger(__name__)
# reader = Reader(['en'])

# servo = Servo(pin=18)

# class PlateRecognizer:

#     def __init__(self,
#                  model_path_1: str = "./model/best-1.pt",
#                  model_path_2: str = "./model/best-2.pt",
#                  video_path_in: str = "./videos/masuk.mp4",
#                  video_path_out: str = "./videos/keluar.mp4"):
#         try:
#             self.model_1 = YOLO(model_path_1)
#             self.model_2 = YOLO(model_path_2)
#         except Exception as e:
#             log.error(f"Gagal memuat model YOLO: {e}")
#             raise e

#         self.cap_in = cv2.VideoCapture(video_path_in)
#         if not self.cap_in.isOpened():
#             log.error(f"Gagal membuka video {video_path_in}")
#             raise IOError(f"Gagal membuka video {video_path_in}")

#         self.cap_out = cv2.VideoCapture(video_path_out)
#         if not self.cap_out.isOpened():
#             log.error(f"Gagal membuka video {video_path_out}")
#             raise IOError(f"Gagal membuka video {video_path_out}")

#         frame_width = int(self.cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(self.cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         desired_width = 720
#         scale = desired_width / frame_width
#         self.new_width = int(frame_width * scale)
#         self.new_height = int(frame_height * scale)

#         self.detection_times_in = defaultdict(float)
#         self.detection_times_out = defaultdict(float)
#         self.data_plates = {}
#         self.lock = threading.Lock()
#         self.state_plates = {}

#         self.detected_text_in = ""
#         self.detected_text_out = ""
#         self.bbox = [0, 0, 0, 0]
#         self.text = ""

#     def run(self):
#         thread_in = threading.Thread(target=self.camera_in)
#         thread_out = threading.Thread(target=self.camera_out)
#         thread_in.start()
#         thread_out.start()
#         thread_in.join()
#         thread_out.join()

#     def camera_in(self):
#         self._process_camera(self.cap_in, "Camera in", self.model_1)

#     def camera_out(self):
#         self._process_camera(self.cap_out, "Camera out", self.model_2)

#     def _process_camera(self, cap, camera_name, model):
#         log.info(f"Memulai pemrosesan video {camera_name}...")

#         fps = 0
#         frame_count = 0
#         start_time = time.time()

#         while cap.isOpened():
#             success, frame = cap.read()

#             if success:
#                 frame_count += 1
#                 if frame_count >= 10:
#                     end_time = time.time()
#                     fps = frame_count / (end_time - start_time)
#                     frame_count = 0
#                     start_time = time.time()

#                 resized_frame = cv2.resize(
#                     frame, (self.new_width, self.new_height))
#                 try:
#                     results = model(resized_frame, verbose=False)
#                 except AttributeError as e:
#                     log.error(f"Error saat memproses frame dengan model: {e}")
#                     break
#                 except Exception as e:
#                     log.error(f"Error tak terduga saat memproses frame: {e}")
#                     continue

#                 current_time = time.time()

#                 detected_text = ""
#                 for r in results:
#                     boxes = r.boxes
#                     for box in boxes:
#                         self.bbox = box.xyxy[0].tolist()
#                         text = self.extract_text(resized_frame, self.bbox)
#                         log.info(f"{camera_name} Detected text: {text}")
#                         detected_text = text

#                         if camera_name == 'Camera in':
#                             self.handle_camera_in(text, current_time)
#                         else:
#                             self.handle_camera_out(text, current_time)

#                 if camera_name == 'Camera in':
#                     self.detected_text_in = detected_text
#                 elif camera_name == 'Camera out':
#                     self.detected_text_out = detected_text

#                 try:
#                     annotated_frame = results[0].plot()
#                     if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
#                         annotated_frame = cv2.cvtColor(
#                             annotated_frame, cv2.COLOR_RGB2BGR)

#                     # tampilkan FPS pada frame
#                     cv2.putText(
#                         annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                     # Hanya gambar teks jika terdeteksi pada kamera yang sesuai
#                     if detected_text:
#                         x1, y1, x2, y2 = map(int, self.bbox)
#                         cv2.putText(annotated_frame, detected_text, (x1, y2 + 20),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#                     cv2.imshow(
#                         f"YOLOv8 Inference {camera_name}", annotated_frame)
#                 except Exception as e:
#                     log.error(f"Error saat menampilkan frame: {e}")

#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     log.info("Video processing stopped by user.")
#                     print("DATA PLATES:", self.data_plates)
#                     print("STATE PLATES:", self.state_plates)
#                     break
#             else:
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         log.info(f"Pemrosesan video {camera_name} selesai.")

#     def handle_camera_in(self, text, current_time):
#         with self.lock:
#             if text not in self.data_plates and text != "":
#                 self.data_plates[text] = current_time
#                 self.state_plates[text] = True
#                 servo.rotate(90)
#                 print(f"Plat '{text}' masuk pada {time.ctime(current_time)}\n")

#     def handle_camera_out(self, text, current_time):
#         with self.lock:
#             if text in self.data_plates:
#                 time_in = self.data_plates[text]
#                 time_out = current_time
#                 duration = time_out - time_in
#                 servo.rotate(90)
#                 print(
#                     f"Kendaraan Plat '{text}' \nmasuk: {time.ctime(time_in)}\nkeluar: {time.ctime(time_out)}.\ndurasi: {round(duration)} detik \n")

#                 del self.data_plates[text]
#                 del self.state_plates[text]

#     def extract_text(self, frame, bbox):
#         x1, y1, x2, y2 = map(int, bbox)
#         roi = frame[y1:y2, x1:x2]
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

#         results = reader.readtext(gray)
#         text = [content for (_, content, _) in results]
#         full_text = ' '.join(text)
#         processed_plates = self.process_plate_text(full_text)

#         return processed_plates

#     def process_plate_text(self, text):
#         lines = text.split('\n')
#         processed_plates = []
#         for line in lines:
#             clean_line = re.sub(r'[^a-zA-Z0-9]', ' ', line)
#             parts = clean_line.split()

#             if len(parts) >= 3:
#                 part1 = ''.join(filter(str.isalpha, parts[0]))[:2].upper()
#                 part2 = ''.join(filter(str.isdigit, ''.join(parts[1:-1])))[:4]
#                 part3 = ''.join(filter(str.isalpha, parts[-1]))[:2].upper()

#                 if not part3:
#                     for part in parts[2:-1]:
#                         part3 = ''.join(filter(str.isalpha, part))[:2].upper()
#                         if part3:
#                             break

#                 if part1 and part2 and part3:
#                     processed_plate = f"{part1} {part2} {part3}"
#                     processed_plates.append(processed_plate)

#         return '\n'.join(processed_plates)

from .servo import Servo

import logging as log
from collections import defaultdict
from ultralytics import YOLO
from easyocr import Reader
import logging
import time
import cv2
import re

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)
reader = Reader(['en'])

servo = Servo(pin=18)


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

        self.video_path_in = video_path_in
        self.video_path_out = video_path_out

        self.data_plates = {}
        self.state_plates = {}

        self.detected_text_in = ""
        self.detected_text_out = ""
        self.bbox = [0, 0, 0, 0]
        self.text = ""

    def run(self):
        self._process_camera(self.video_path_in, "Camera in", self.model_1)
        self._process_camera(self.video_path_out, "Camera out", self.model_2)

    def _process_camera(self, video_path, camera_name, model):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error(f"Gagal membuka video {video_path}")
            return

        log.info(f"Memulai pemrosesan video {camera_name}...")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        desired_width = 720
        scale = desired_width / frame_width
        self.new_width = int(frame_width * scale)
        self.new_height = int(frame_height * scale)

        fps = 0
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                frame_count += 1
                if frame_count >= 10:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = time.time()

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

                detected_text = ""
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        self.bbox = box.xyxy[0].tolist()
                        text = self.extract_text(resized_frame, self.bbox)
                        log.info(f"{camera_name} Detected text: {text}")
                        detected_text = text

                        if camera_name == 'Camera in':
                            self.handle_camera_in(text, current_time)
                        else:
                            self.handle_camera_out(text, current_time)

                if camera_name == 'Camera in':
                    self.detected_text_in = detected_text
                elif camera_name == 'Camera out':
                    self.detected_text_out = detected_text

                try:
                    annotated_frame = results[0].plot()
                    if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                        annotated_frame = cv2.cvtColor(
                            annotated_frame, cv2.COLOR_RGB2BGR)

                    cv2.putText(
                        annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if detected_text:
                        x1, y1, x2, y2 = map(int, self.bbox)
                        cv2.putText(annotated_frame, detected_text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
        if text not in self.data_plates and text != "":
            self.data_plates[text] = current_time
            self.state_plates[text] = True
            servo.rotate(90)
            print(f"Plat '{text}' masuk pada {time.ctime(current_time)}\n")

    def handle_camera_out(self, text, current_time):
        if text in self.data_plates:
            time_in = self.data_plates[text]
            time_out = current_time
            duration = time_out - time_in
            servo.rotate(90)
            print(
                f"Kendaraan Plat '{text}' \nmasuk: {time.ctime(time_in)}\nkeluar: {time.ctime(time_out)}.\ndurasi: {round(duration)} detik \n")

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
