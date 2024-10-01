

class Servo:
    def __init__(self, pin):
        self.pin = pin

    def rotate(self, angle):
        print(f"Servo on pin {self.pin} rotated to {angle} degrees")