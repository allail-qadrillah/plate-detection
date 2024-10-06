from src import PlateRecognizer

if __name__ == "__main__":
    # Initialize the PlateRecognizer object
    plate_recognizer = PlateRecognizer(
        video_path_in="./videos/cepat-masuk.mp4",
        video_path_out="./videos/cepat-keluar.mp4"
    )

    # Run the PlateRecognizer
    plate_recognizer.run()


# https: //www.instructables.com/Controlling-Servo-Motor-Sg90-With-Raspberry-Pi-4/