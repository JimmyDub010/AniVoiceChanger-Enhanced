"""Bounded, non-executing reader for ordinary RVC tensor checkpoints.

No pickle.loads/Unpickler/torch.load is used. GLOBAL/REDUCE are interpreted as
inert tags for a very small tensor/OrderedDict grammar, never invoked.
"""
from dataclasses import dataclass
import io
import math
from pathlib import Path, PurePosixPath
import pickletools
import re
import struct
from zipfile import ZipFile, is_zipfile

MAX_FILE = 1024 ** 3
MAX_PICKLE = 8 * 1024 ** 2
MAX_OPS = 250000
MAX_STORAGE = 512 * 1024 ** 2
STORAGE_TYPES = {'HalfStorage': ('<f2', 2), 'FloatStorage': ('<f4', 4),
                 'DoubleStorage': ('<f8', 8), 'LongStorage': ('<i8', 8),
                 'IntStorage': ('<i4', 4), 'ShortStorage': ('<i2', 2),
                 'ByteStorage': ('u1', 1), 'BoolStorage': ('?', 1)}


@dataclass(frozen=True)
class Global:
    module: str
    name: str


@dataclass(frozen=True)
class Storage:
    key: str
    kind: str
    count: int


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


class CheckpointError(ValueError):
    pass


def valid_member(name):
    name = name.replace('\\', '/')
    path = PurePosixPath(name)
    if not name or len(name) > 512 or path.is_absolute() or ':' in name or any(x in ('.', '..') for x in name.split('/')):
        raise CheckpointError('Unsafe archive path.')
    if any(ord(c) < 32 for c in name):
        raise CheckpointError('Invalid archive filename.')
    return name


class TensorGrammar:
    def __init__(self):
        self.storages = {}

    def parse(self, stream):
        start = stream.tell()
        original = stream
        class Limited:
            def read(self, size=-1):
                remaining = MAX_PICKLE - (original.tell() - start)
                if size < 0 or size > remaining: raise CheckpointError('Checkpoint metadata read exceeds its limit.')
                return original.read(size)
            def readline(self):
                value = original.readline(MAX_PICKLE - (original.tell() - start) + 1)
                if original.tell() - start > MAX_PICKLE: raise CheckpointError('Checkpoint metadata exceeds its limit.')
                return value
            def tell(self): return original.tell()
        stream = Limited()
        stack, memo = [], {}
        mark = object()

        def pop_mark():
            for i in range(len(stack) - 1, -1, -1):
                if stack[i] is mark:
                    values = stack[i + 1:]
                    del stack[i:]
                    return values
            raise CheckpointError('Invalid pickle stack.')

        def global_tag(module, name):
            allowed = ((module == 'collections' and name == 'OrderedDict') or
                       (module == 'torch' and name in STORAGE_TYPES) or
                       (module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_parameter')))
            if not allowed:
                raise CheckpointError('Unsupported executable checkpoint global: ' + module + '.' + name)
            return Global(module, name)

        for number, (op, arg, position) in enumerate(pickletools.genops(stream)):
            if number > MAX_OPS or stream.tell() - start > MAX_PICKLE or len(stack) > 50000 or len(memo) > 100000:
                raise CheckpointError('Checkpoint metadata exceeds validation limits.')
            name = op.name
            if name in ('PROTO', 'FRAME'):
                continue
            if name == 'MARK': stack.append(mark)
            elif name in ('EMPTY_DICT', 'EMPTY_LIST', 'EMPTY_TUPLE'):
                stack.append({} if name == 'EMPTY_DICT' else [] if name == 'EMPTY_LIST' else ())
            elif name in ('NONE', 'NEWTRUE', 'NEWFALSE'):
                stack.append(None if name == 'NONE' else name == 'NEWTRUE')
            elif name in ('BININT', 'BININT1', 'BININT2', 'INT', 'LONG', 'LONG1', 'LONG4', 'FLOAT', 'BINFLOAT'):
                if isinstance(arg, (int, float)) and abs(arg) > 2 ** 128:
                    raise CheckpointError('Invalid numeric metadata.')
                stack.append(arg)
            elif name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE', 'BINUNICODE8', 'SHORT_BINSTRING', 'BINSTRING'):
                if len(arg) > 1024 ** 2: raise CheckpointError('Oversized checkpoint string.')
                stack.append(arg.decode('utf-8') if isinstance(arg, bytes) else arg)
            elif name == 'GLOBAL':
                stack.append(global_tag(*arg.split(' ', 1)))
            elif name == 'STACK_GLOBAL':
                target, module = stack.pop(), stack.pop()
                stack.append(global_tag(module, target))
            elif name in ('BINPUT', 'LONG_BINPUT', 'PUT'):
                if int(arg) > 100000: raise CheckpointError('Oversized checkpoint memo.')
                memo[int(arg)] = stack[-1]
            elif name == 'MEMOIZE': memo[len(memo)] = stack[-1]
            elif name in ('BINGET', 'LONG_BINGET', 'GET'): stack.append(memo[int(arg)])
            elif name == 'TUPLE': stack.append(tuple(pop_mark()))
            elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                n = int(name[-1]); values = tuple(stack[-n:]); del stack[-n:]; stack.append(values)
            elif name == 'LIST': stack.append(pop_mark())
            elif name == 'DICT':
                values = pop_mark(); stack.append(dict(zip(values[::2], values[1::2])))
            elif name == 'APPENDS':
                values = pop_mark(); stack[-1].extend(values)
            elif name == 'APPEND':
                value = stack.pop(); stack[-1].append(value)
            elif name == 'SETITEMS':
                values = pop_mark()
                if len(values) % 2: raise CheckpointError('Invalid dictionary metadata.')
                stack[-1].update(zip(values[::2], values[1::2]))
            elif name == 'SETITEM':
                value, key = stack.pop(), stack.pop(); stack[-1][key] = value
            elif name == 'BINPERSID':
                value = stack.pop()
                if not isinstance(value, tuple) or len(value) not in (5, 6) or value[0] != 'storage':
                    raise CheckpointError('Unsupported checkpoint storage.')
                _, kind, key, location, count, *extra = value
                if (not isinstance(kind, Global) or kind.module != 'torch' or kind.name not in STORAGE_TYPES
                        or not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,80}', key)
                        or not isinstance(count, int) or count < 0 or count * STORAGE_TYPES[kind.name][1] > MAX_STORAGE
                        or not isinstance(location, str) or not re.fullmatch(r'cpu|cuda(?::\d+)?', location)
                        or (extra and extra[0] is not None)):
                    raise CheckpointError('Unsupported or oversized tensor storage.')
                value = Storage(key, kind.name, count)
                if key in self.storages and self.storages[key] != value:
                    raise CheckpointError('Inconsistent tensor storage.')
                self.storages[key] = value
                stack.append(value)
            elif name == 'REDUCE':
                args, function = stack.pop(), stack.pop()
                if not isinstance(function, Global) or not isinstance(args, tuple):
                    raise CheckpointError('Unsupported checkpoint operation.')
                if function == Global('collections', 'OrderedDict') and not args:
                    stack.append({})
                elif function.module == 'torch._utils' and function.name in ('_rebuild_tensor', '_rebuild_tensor_v2'):
                    storage, offset, shape, stride = args[:4]
                    if (not isinstance(storage, Storage) or not isinstance(offset, int) or offset < 0
                            or not isinstance(shape, tuple) or not isinstance(stride, tuple)
                            or len(shape) != len(stride) or len(shape) > 8
                            or any(not isinstance(n, int) or n < 0 or n > 10000000 for n in shape + stride)
                            or math.prod(shape) > 100000000):
                        raise CheckpointError('Invalid or oversized tensor layout.')
                    last = offset + sum((n - 1) * step for n, step in zip(shape, stride) if n)
                    if math.prod(shape) and last >= storage.count:
                        raise CheckpointError('Tensor layout exceeds storage.')
                    stack.append(Tensor(storage, offset, shape, stride))
                elif function == Global('torch._utils', '_rebuild_parameter') and isinstance(args[0], Tensor):
                    stack.append(args[0])
                else:
                    raise CheckpointError('Unsupported checkpoint constructor.')
            elif name == 'STOP':
                if len(stack) != 1: raise CheckpointError('Invalid checkpoint metadata stack.')
                return stack[0]
            else:
                raise CheckpointError('Unsupported checkpoint instruction: ' + name)
        raise CheckpointError('Incomplete checkpoint metadata.')


class SafeCheckpoint:
    def __init__(self, file):
        self.path = Path(file)
        if not self.path.is_file() or not 1 <= self.path.stat().st_size <= MAX_FILE:
            raise CheckpointError('Model is missing or exceeds the 1 GiB model limit.')
        grammar = TensorGrammar()
        self.members, self.offsets = {}, {}
        self.zipped = is_zipfile(self.path)
        if self.zipped:
            with ZipFile(self.path) as archive:
                entries = archive.infolist()
                if len(entries) > 10000: raise CheckpointError('Too many checkpoint entries.')
                total = 0
                for entry in entries:
                    name = valid_member(entry.filename)
                    if name in self.members or entry.flag_bits & 1 or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                        raise CheckpointError('Unsafe checkpoint archive entry.')
                    total += entry.file_size
                    if total > MAX_FILE or entry.file_size > MAX_STORAGE or entry.file_size > max(1, entry.compress_size) * 200:
                        raise CheckpointError('Oversized or excessive checkpoint compression.')
                    self.members[name] = entry
                pickles = [name for name in self.members if name.endswith('/data.pkl') or name == 'data.pkl']
                if len(pickles) != 1 or self.members[pickles[0]].file_size > MAX_PICKLE:
                    raise CheckpointError('Not an ordinary PyTorch tensor checkpoint.')
                self.prefix = pickles[0][:-len('data.pkl')]
                self.document = grammar.parse(io.BytesIO(archive.read(pickles[0])))
                for storage in grammar.storages.values():
                    name = self.prefix + 'data/' + storage.key
                    if name not in self.members or self.members[name].file_size != storage.count * STORAGE_TYPES[storage.kind][1]:
                        raise CheckpointError('Tensor storage is missing or has an invalid size.')
                byteorder = self.prefix + 'byteorder'
                if byteorder in self.members and archive.read(byteorder) != b'little':
                    raise CheckpointError('Only little-endian checkpoints are supported.')
        else:
            with self.path.open('rb') as stream:
                if grammar.parse(stream) != 0x1950A86A20F9469CFC6C or grammar.parse(stream) != 1001:
                    raise CheckpointError('Not a supported legacy tensor checkpoint.')
                system = grammar.parse(stream)
                if not isinstance(system, dict) or system.get('little_endian') is not True:
                    raise CheckpointError('Unsupported checkpoint byte order.')
                self.document = grammar.parse(stream)
                keys = grammar.parse(stream)
                if not isinstance(keys, list) or set(keys) != set(grammar.storages) or len(keys) != len(set(keys)):
                    raise CheckpointError('Invalid legacy storage table.')
                for key in keys:
                    storage = grammar.storages[key]
                    count = struct.unpack('<q', stream.read(8))[0]
                    size = storage.count * STORAGE_TYPES[storage.kind][1]
                    if count != storage.count or stream.tell() + size > self.path.stat().st_size:
                        raise CheckpointError('Invalid legacy tensor data.')
                    self.offsets[key] = (stream.tell(), size)
                    stream.seek(size, 1)
                if stream.tell() != self.path.stat().st_size:
                    raise CheckpointError('Unexpected trailing checkpoint data.')
        self.storages = grammar.storages
        if sum(s.count * STORAGE_TYPES[s.kind][1] for s in self.storages.values()) > MAX_FILE:
            raise CheckpointError('Excessive tensor storage.')
        self.metadata = self.validate_rvc()

    def validate_rvc(self):
        checkpoint = self.document
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('weight'), dict):
            raise CheckpointError('This is not an RVC inference model: weight/config are required.')
        config = checkpoint.get('config')
        if not isinstance(config, (list, tuple)) or len(config) != 18:
            raise CheckpointError('Invalid RVC architecture configuration.')
        version = checkpoint.get('version', 'v1')
        rate = config[-1]
        f0 = checkpoint.get('f0', 1)
        if version not in ('v1', 'v2') or rate not in (32000, 40000, 48000) or f0 not in (0, 1):
            raise CheckpointError('Unsupported RVC version, sample rate or pitch flag.')
        def primitive(value, depth=0):
            if depth > 5: raise CheckpointError('Invalid nested RVC config.')
            if isinstance(value, (int, float, str, bool)):
                if isinstance(value, str) and len(value) > 100: raise CheckpointError('Invalid config string.')
                if isinstance(value, (int, float)) and (not math.isfinite(value) or abs(value) > 1000000):
                    raise CheckpointError('Invalid config value.')
            elif isinstance(value, (list, tuple)) and len(value) <= 100:
                for item in value: primitive(item, depth + 1)
            else: raise CheckpointError('Invalid RVC config value.')
        primitive(config)
        bounds = {0: (1, 4097), 1: (1, 2048), 2: (16, 512), 3: (16, 512),
                  4: (16, 2048), 5: (1, 8), 6: (1, 24), 7: (1, 15),
                  13: (16, 1024), 15: (1, 4096), 16: (1, 512)}
        if any(type(config[i]) is not int or not low <= config[i] <= high for i, (low, high) in bounds.items()):
            raise CheckpointError('Architecture sizes exceed supported RVC limits.')
        if not isinstance(config[8], (int, float)) or not 0 <= config[8] <= 1 or config[9] not in ('1', '2'):
            raise CheckpointError('Unsupported RVC dropout or resblock configuration.')
        for values in (config[10], config[12], config[14]):
            if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 8 or any(type(v) is not int or not 1 <= v <= 64 for v in values):
                raise CheckpointError('Unsupported convolution configuration.')
        if (len(config[12]) != len(config[14]) or math.prod(config[12]) * 100 != rate
                or not isinstance(config[11], (list, tuple)) or len(config[11]) != len(config[10])
                or any(not isinstance(v, (list, tuple)) or not 1 <= len(v) <= 8
                       or any(type(n) is not int or not 1 <= n <= 64 for n in v) for v in config[11])):
            raise CheckpointError('Invalid RVC upsampling or dilation configuration.')
        weights = checkpoint['weight']
        if not 2 <= len(weights) <= 5000 or not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in weights.items()):
            raise CheckpointError('Unsupported RVC tensor dictionary.')
        emb, phone = weights.get('emb_g.weight'), weights.get('enc_p.emb_phone.weight')
        dimensions = 256 if version == 'v1' else 768
        if (not isinstance(emb, Tensor) or len(emb.shape) != 2 or not 1 <= emb.shape[0] <= 4096
                or emb.shape[1] != config[16] or not isinstance(phone, Tensor) or len(phone.shape) != 2
                or phone.shape[1] != dimensions or phone.shape[0] != config[3]):
            raise CheckpointError('Model feature tensors do not match its declared RVC version.')
        return {'version': version, 'sample_rate': rate, 'f0': int(f0), 'dimensions': dimensions,
                'tensor_count': len(weights), 'validation': 'non-executing tensor grammar; architecture plausibility'}

    def materialize(self):
        import numpy as np
        import torch
        bases = {}
        archive = ZipFile(self.path) if self.zipped else None
        try:
            with self.path.open('rb') as stream:
                for key, storage in self.storages.items():
                    if archive:
                        raw = archive.read(self.prefix + 'data/' + key)
                    else:
                        offset, size = self.offsets[key]; stream.seek(offset); raw = stream.read(size)
                    bases[key] = torch.from_numpy(np.frombuffer(raw, dtype=STORAGE_TYPES[storage.kind][0]).copy())
            weights = {name: torch.as_strided(bases[value.storage.key], value.shape, value.stride, value.offset)
                       for name, value in self.document['weight'].items()}
            return {'config': self.document['config'], 'weight': weights,
                    'version': self.metadata['version'], 'f0': self.metadata['f0']}
        finally:
            if archive: archive.close()

    def verify_payloads(self):
        if not self.zipped: return
        # Force ZIP CRC validation for every declared tensor without retaining it in RAM.
        with ZipFile(self.path) as archive:
            for key in self.storages:
                with archive.open(self.prefix + 'data/' + key) as stream:
                    while stream.read(1024 ** 2): pass
