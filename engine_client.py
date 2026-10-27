"""Keep legacy RVC dependencies isolated in the downloaded portable runtime."""
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
from threading import Lock, Thread

ROOT = Path(__file__).resolve().parent


class EngineClient:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        python = self.folder / 'runtime' / 'python.exe'
        if not python.is_file():
            raise ValueError('The selected RVC installation has no runtime/python.exe.')
        self.lock = Lock()
        self.responses = Queue()
        cache = ROOT / '.cache'
        cache.mkdir(exist_ok=True)
        self.log = (cache / 'engine.log').open('a', encoding='utf-8')
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                           NUMBA_CACHE_DIR=str(cache / 'numba'),
                           MPLCONFIGDIR=str(cache / 'matplotlib'))
        self.process = subprocess.Popen(
            [str(python), '-B', '-u', str(ROOT / 'engine_worker.py'), str(self.folder)],
            cwd=self.folder, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW)
        def read():
            for line in self.process.stdout:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.responses.put({'ok': False, 'error': 'Unexpected output from RVC worker. See .cache/engine.log.'})
            self.responses.put({'ok': False, 'error': 'RVC worker exited. See .cache/engine.log.'})
        Thread(target=read, daemon=True).start()
        if not self.receive(30).get('ready'):
            self.close()
            raise RuntimeError('RVC worker did not become ready.')

    def receive(self, timeout):
        try:
            return self.responses.get(timeout=timeout)
        except Empty:
            self.close()
            raise TimeoutError('RVC operation timed out. See .cache/engine.log.')

    def request(self, command, **values):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError('RVC worker is no longer running. Load the model again.')
            self.process.stdin.write(json.dumps(dict(values, command=command)) + '\n')
            self.process.stdin.flush()
            response = self.receive(180)
            if not response.get('ok'):
                raise RuntimeError(response.get('error', 'Unknown RVC worker error'))
            return response['result']

    def close(self):
        if hasattr(self, 'process') and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if hasattr(self, 'process'):
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe:
                    pipe.close()
        if hasattr(self, 'log'):
            self.log.close()
