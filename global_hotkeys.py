"""Global key bridge and independently testable capture state machine."""
import ctypes
import time


class CaptureKeys:
    def __init__(self, start, stop, ready, announce, physically_down, clock=time.monotonic):
        self.start, self.stop, self.ready = start, stop, ready
        self.announce, self.physically_down, self.clock = announce, physically_down, clock
        self.enabled = False
        self.toggle = False
        self.down = False
        self.recording = False
        self.started = 0
        self.release_misses = 0

    def enable(self, toggle=False):
        self.toggle = bool(toggle)
        self.enabled = True
        self.down = False
        self.release_misses = 0

    def disable(self):
        self.enabled = False
        self.down = False
        if self.recording:
            self.recording = False
            self.stop('Hotkey disabled.')

    def event(self, event_type):
        if not self.enabled: return
        if event_type == 'down':
            if self.down: return
            self.down = True
            if self.toggle and self.recording:
                self.recording = False
                self.stop('Toggle recording stopped.')
            elif self.ready():
                self.recording = bool(self.start())
                self.started = self.clock()
                self.release_misses = 0
        elif event_type == 'up':
            self.down = False
            self.release_misses = 0
            if self.recording and not self.toggle:
                self.recording = False
                self.stop('Push-to-talk key released.')

    def tick(self):
        if not self.enabled: return
        # Poll real Windows key state, rather than the hook's potentially stale cache.
        if self.down and not self.physically_down():
            self.release_misses += 1
            if self.release_misses >= 2:
                self.down = False
                if self.recording and not self.toggle:
                    self.recording = False
                    self.stop('Recovered a missed key-release event.')
        else:
            self.release_misses = 0
        if self.recording and self.clock() - self.started >= 59:
            self.recording = False
            # Keep the held-key latch until release so auto-repeat cannot restart.
            self.stop('Recording safety limit reached.')


class GlobalKeyBridge:
    def __init__(self, controller, dispatch):
        self.controller = controller
        self.dispatch = dispatch
        self.hooks = []
        self.vk = None

    def enable(self, key, toggle=False):
        import keyboard
        self.disable()
        key = key.strip().lower()
        if not key or '+' in key or len(key) > 30:
            raise ValueError('Choose one global key, such as f8. Key combinations are not supported.')
        scans = keyboard.key_to_scan_codes(key)
        if len(scans) != 1:
            raise ValueError('Choose an unambiguous single key, such as f8.')
        self.vk = ctypes.windll.user32.MapVirtualKeyW(scans[0], 1)
        if not self.vk: raise ValueError('This key cannot be monitored reliably on Windows.')
        try:
            self.hooks.append(keyboard.on_press_key(key, lambda event: self.dispatch(self.controller.event, 'down'), suppress=False))
            self.hooks.append(keyboard.on_release_key(key, lambda event: self.dispatch(self.controller.event, 'up'), suppress=False))
            self.controller.enable(toggle)
        except Exception:
            self.disable()
            raise

    def physically_down(self):
        return bool(self.vk and ctypes.windll.user32.GetAsyncKeyState(self.vk) & 0x8000)

    def disable(self):
        import keyboard
        for hook in self.hooks:
            keyboard.unhook(hook)
        self.hooks = []
        self.controller.disable()
