"""Generation tokens invalidate audio without interrupting an in-use model."""
from threading import Event, Lock


class Generations:
    def __init__(self):
        self.number = 0
        self.cancel = Event()
        self.lock = Lock()

    def advance(self):
        with self.lock:
            self.cancel.set()
            self.number += 1
            self.cancel = Event()
            return self.number

    def snapshot(self):
        with self.lock:
            return self.number, self.cancel

    def valid(self, number):
        with self.lock:
            return number == self.number and not self.cancel.is_set()
