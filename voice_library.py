"""Atomic, relocatable RVC library. Source archives/files are read-only inputs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from safe_checkpoint import SafeCheckpoint, valid_member

MAX_ENTRIES = 2048
MAX_FILE = 1024 ** 3
MAX_TOTAL = 2 * 1024 ** 3
MAX_RATIO = 200
AUTO = 'automatic'
CATEGORIES = ['Singer', 'Wrestler', 'Character', 'Announcer', 'Other']
USAGES = ['Talking', 'Singing', 'Both', 'Unknown']
METHODS = ['rmvpe', 'crepe', 'rmvpe-new', 'pm', 'harvest', 'fcpe-experimental']


class SelectionRequired(ValueError):
    def __init__(self, kind, choices, model=None):
        super().__init__('Choose the ' + kind + ' for this voice.')
        self.kind, self.choices, self.model = kind, choices, model


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp-' + uuid4().hex)
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def index_header(stream, dimensions):
    header = stream.read(40)
    if len(header) < 37 or header[:4] not in (b'IwFl', b'IxF2', b'IxFI', b'IxPQ', b'IwPQ'):
        raise ValueError('Unsupported or corrupted FAISS index header.')
    dimension, count = struct.unpack('<i', header[4:8])[0], struct.unpack('<q', header[8:16])[0]
    if dimension != dimensions:
        raise ValueError(f'Index has {dimension} features; this model needs {dimensions}.')
    if not 8 <= count <= 2000000 or count * dimension * 4 > 512 * 1024 ** 2 or header[32] != 1:
        raise ValueError('Index is empty, untrained, or exceeds the supported vector limit.')
    return {'dimensions': dimension, 'vectors': count,
            'validation': 'bounded FAISS header and dimensional compatibility; native inference validates structure'}


def archive_catalog(path):
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ENTRIES: raise ValueError('ZIP has too many entries.')
            seen, total, models, indexes = set(), 0, [], []
            for entry in entries:
                name = valid_member(entry.filename.rstrip('/'))
                key = name.casefold()
                mode = entry.external_attr >> 16
                if key in seen: raise ValueError('ZIP contains duplicate or colliding filenames.')
                seen.add(key)
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError('ZIP links and special files are not allowed.')
                if entry.flag_bits & 1: raise ValueError('Encrypted ZIP entries are not supported.')
                total += entry.file_size
                if entry.file_size > MAX_FILE or total > MAX_TOTAL or entry.file_size > max(1, entry.compress_size) * MAX_RATIO:
                    raise ValueError('ZIP exceeds extraction size or compression limits.')
                if entry.is_dir(): continue
                suffix = Path(name).suffix.lower()
                if suffix == '.pth': models.append(entry.filename)
                if suffix == '.index': indexes.append(entry.filename)
            if not models: raise ValueError('ZIP contains no .pth voice model.')
            return {'models': models, 'indexes': indexes}
    except BadZipFile as error:
        raise ValueError('The ZIP archive is corrupted or incomplete.') from error


def pair_score(model, index):
    model_path, index_path = Path(model.replace('\\', '/')), Path(index.replace('\\', '/'))
    stem = re.sub('[^a-z0-9]', '', model_path.stem.lower())
    other = re.sub('[^a-z0-9]', '', index_path.stem.lower())
    score = 0
    if stem == other: score += 100
    if len(stem) > 3 and stem not in ('model', 'weights') and stem in other: score += 60
    if model_path.parent == index_path.parent: score += 20
    if index_path.name.lower().startswith('added_'): score += 2
    return score


class VoiceLibrary:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'voices.json'
        self.backup = self.root / 'voices.json.bak'
        self.recovered = False
        self.warning = ''
        with self.lock():
            if not self.path.exists() and not self.backup.exists():
                self.write({'schema': 1, 'voices': [], 'removed': [], 'last_voice': None})
            self.read()

    @contextmanager
    def lock(self):
        import msvcrt
        with (self.root / '.registry.lock').open('a+b') as stream:
            if stream.tell() == 0: stream.write(b'0'); stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError('The voice library is busy in another application instance.') from error
            try: yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

    def validate(self, doc):
        if (not isinstance(doc, dict) or doc.get('schema') != 1 or not isinstance(doc.get('voices'), list)
                or not isinstance(doc.get('removed'), list) or len(doc['voices']) + len(doc['removed']) > 10000):
            raise ValueError('Invalid voice registry schema.')
        ids = set()
        for voice in doc['voices'] + doc['removed']:
            if not isinstance(voice, dict) or not re.fullmatch('[0-9a-f]{32}', voice.get('id', '')) or voice['id'] in ids:
                raise ValueError('Invalid or duplicate voice ID.')
            ids.add(voice['id'])
            if not isinstance(voice.get('name'), str) or not 1 <= len(voice['name']) <= 200:
                raise ValueError('Invalid voice name.')
            directory = voice.get('directory_id', voice['id'])
            if not re.fullmatch('[0-9a-f]{32}', directory): raise ValueError('Invalid voice directory.')
            for key, suffix in [('model', 'model.pth'), ('index', 'model.index')]:
                if key == 'index' and voice.get(key) is None: continue
                if voice.get(key) != f'voices/{directory}/{suffix}': raise ValueError('Unsafe voice registry path.')
            self.valid_preferences(voice.get('preferences', {}))
            if voice.get('category', 'Other') not in CATEGORIES or voice.get('usage', 'Unknown') not in USAGES:
                raise ValueError('Invalid voice category or usage.')
        if doc.get('last_voice') is not None and doc['last_voice'] not in [v['id'] for v in doc['voices']]:
            raise ValueError('Invalid last voice selection.')
        return doc

    @staticmethod
    def valid_preferences(value):
        if not isinstance(value, dict): raise ValueError('Invalid voice preferences.')
        if (type(value.get('pitch', 12)) is not int or not -24 <= value.get('pitch', 12) <= 24
                or value.get('method', 'rmvpe') not in METHODS
                or type(value.get('index_rate', 0)) not in (int, float)
                or not 0 <= value.get('index_rate', 0) <= 1 or not isinstance(value.get('effects', {}), dict)):
            raise ValueError('Invalid pitch, method, index influence or effects.')
        from audio_effects import settings
        settings(value.get('effects'))

    def read_file(self, path):
        if path.stat().st_size > 8 * 1024 ** 2: raise ValueError('Voice registry is too large.')
        return self.validate(json.loads(path.read_text(encoding='utf-8')))

    def read(self):
        try:
            return self.read_file(self.path)
        except (OSError, ValueError, TypeError, KeyError):
            try:
                document = self.read_file(self.backup)
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise ValueError('Both voice registry and recovery copy are unavailable or invalid. Original voice files were not changed.') from error
            if self.path.exists():
                preserved = self.root / ('voices-corrupt-' + uuid4().hex + '.json')
                shutil.copyfile(self.path, preserved)
            atomic_json(self.path, document)
            self.recovered = True
            self.warning = 'Voice registry recovered from its backup. The corrupt file was preserved.'
            return document

    def write(self, doc):
        self.validate(doc)
        if self.path.exists():
            try:
                old = self.read_file(self.path)
            except (OSError, ValueError, TypeError, KeyError):
                old = None
            if old is not None: atomic_json(self.backup, old)
        atomic_json(self.path, doc)
        # Primary replacement is the commit point. A later backup error must not
        # roll back files after a valid registry already references them.
        try: atomic_json(self.backup, doc)
        except OSError:
            self.warning = 'Registry saved, but its recovery copy could not be refreshed.'

    def voices(self, include_removed=False):
        with self.lock():
            doc = self.read()
            return doc['voices'] + (doc['removed'] if include_removed else [])

    def get(self, voice_id):
        return next(v for v in self.voices() if v['id'] == voice_id)

    def path_for(self, voice, key):
        if not voice.get(key): return None
        path = (self.root / voice[key]).resolve()
        if not path.is_relative_to(self.root): raise ValueError('Voice path escapes the library.')
        return path

    def selected(self):
        with self.lock(): return self.read().get('last_voice')

    def select(self, voice_id):
        with self.lock():
            doc = self.read(); doc['last_voice'] = voice_id; self.write(doc)

    def import_zip(self, source, model_choice=None, index_choice=AUTO, name=None, preferences=None):
        catalog = archive_catalog(source)
        if model_choice is None:
            if len(catalog['models']) != 1: raise SelectionRequired('model', catalog['models'])
            model_choice = catalog['models'][0]
        if model_choice not in catalog['models']: raise ValueError('Selected model is not in this ZIP.')
        with ZipFile(source) as archive:
            def copy_member(member, target):
                entry = archive.getinfo(member)
                with archive.open(entry) as stream, target.open('xb') as output:
                    self.copy_limited(stream, output, entry.file_size)
            def compatible(dimensions):
                choices = []
                for candidate in catalog['indexes']:
                    try:
                        with archive.open(candidate) as stream: index_header(stream, dimensions)
                        choices.append(candidate)
                    except ValueError:
                        pass
                return choices
            if not name and Path(model_choice).stem.lower() in ('model', 'weights', 'voice', 'checkpoint'):
                name = Path(source).stem
            result = self._import(copy_member, model_choice, index_choice, name, preferences,
                                  compatible, Path(source).name)
            if catalog['indexes'] and not result['voice'].get('index'):
                result['warning'] = 'No compatible usable index was imported. Index influence is off.'
            return result

    def import_files(self, model, index=None, name=None, preferences=None, relink_id=None):
        model = Path(model).resolve()
        index = Path(index).resolve() if index else None
        for file, suffix in [(model, '.pth')] + ([(index, '.index')] if index else []):
            if not file.is_file() or file.suffix.lower() != suffix or file.stat().st_size > MAX_FILE:
                raise ValueError('Choose an existing supported model/index file within the size limit.')
        mapping = {str(model): model}
        if index: mapping[str(index)] = index
        def copy_file(member, target):
            with mapping[member].open('rb') as reader, target.open('xb') as writer:
                self.copy_limited(reader, writer, mapping[member].stat().st_size)
        def compatible(dimensions):
            if index:
                with index.open('rb') as stream: index_header(stream, dimensions)
                return [str(index)]
            return []
        return self._import(copy_file, str(model), str(index) if index else None, name, preferences,
                            compatible, None, relink_id)

    @staticmethod
    def copy_limited(reader, writer, expected):
        count = 0
        for block in iter(lambda: reader.read(1024 ** 2), b''):
            count += len(block)
            if count > min(expected, MAX_FILE): raise ValueError('Extracted file exceeds declared limits.')
            writer.write(block)
        if count != expected: raise ValueError('Incomplete extracted file.')
        writer.flush(); os.fsync(writer.fileno())

    def _cleanup(self, directory):
        directory = directory.resolve()
        if directory.parent not in (self.root, self.root / 'voices') or not directory.is_relative_to(self.root):
            raise ValueError('Refusing cleanup outside the voice library.')
        if directory.is_dir(): shutil.rmtree(directory)

    def _import(self, copier, model_name, index_choice, name, preferences, compatible, archive_name, relink_id=None):
        directory_id = uuid4().hex
        staging = self.root / ('.import-' + directory_id)
        staging.mkdir(exist_ok=False)
        final = self.root / 'voices' / directory_id
        registered = False
        try:
            copier(model_name, staging / 'model.pth')
            model = SafeCheckpoint(staging / 'model.pth')
            model.verify_payloads()
            candidates = compatible(model.metadata['dimensions'])
            if index_choice == AUTO:
                if len(candidates) == 1: index_choice = candidates[0]
                elif len(candidates) > 1:
                    ranked = sorted(candidates, key=lambda x: pair_score(model_name, x), reverse=True)
                    if pair_score(model_name, ranked[0]) >= 20 and pair_score(model_name, ranked[0]) > pair_score(model_name, ranked[1]):
                        index_choice = ranked[0]
                    else: raise SelectionRequired('index (or no index)', candidates + [None], model_name)
                else: index_choice = None
            if index_choice is not None and index_choice not in candidates:
                raise ValueError('Selected index is missing, untrained, corrupted or dimensionally incompatible.')
            index_meta = None
            if index_choice:
                copier(index_choice, staging / 'model.index')
                with (staging / 'model.index').open('rb') as stream: index_meta = index_header(stream, model.metadata['dimensions'])
            model_hash = file_hash(staging / 'model.pth')
            index_hash = file_hash(staging / 'model.index') if index_choice else None
            with self.lock():
                doc = self.read()
                duplicate = next((v for v in doc['voices'] if v['id'] != relink_id and v.get('model_hash') == model_hash and v.get('index_hash') == index_hash), None)
                if duplicate:
                    return {'voice': duplicate, 'duplicate': True}
                prefs = preferences or {'pitch': 12 if model.metadata['f0'] else 0, 'method': 'rmvpe',
                                        'index_rate': 0.75 if index_choice else 0, 'effects': {}}
                if not index_choice: prefs = dict(prefs, index_rate=0)
                if not model.metadata['f0']: prefs = dict(prefs, pitch=0)
                self.valid_preferences(prefs)
                voice = {'id': relink_id or directory_id, 'directory_id': directory_id,
                         'name': (name or Path(model_name.replace('\\', '/')).stem)[:200],
                         'model': f'voices/{directory_id}/model.pth',
                         'index': f'voices/{directory_id}/model.index' if index_choice else None,
                         'original_model_name': Path(model_name.replace('\\', '/')).name,
                         'original_index_name': Path(index_choice.replace('\\', '/')).name if index_choice else None,
                         'archive_member_model': model_name if archive_name else None,
                         'archive_member_index': index_choice if archive_name else None,
                         'source_archive': archive_name, 'model_hash': model_hash, 'index_hash': index_hash,
                         'rvc': model.metadata, 'index_validation': index_meta,
                         'category': 'Other', 'usage': 'Unknown', 'preferences': prefs,
                         'imported_at': datetime.now(timezone.utc).isoformat()}
                if relink_id:
                    old = next(v for v in doc['voices'] if v['id'] == relink_id)
                    for key in ('name', 'category', 'usage', 'preferences', 'imported_at'): voice[key] = old[key]
                    doc['voices'] = [voice if v['id'] == relink_id else v for v in doc['voices']]
                else: doc['voices'].append(voice)
                final.parent.mkdir(exist_ok=True)
                staging.rename(final)
                try: self.write(doc)
                except Exception:
                    self._cleanup(final)
                    raise
                registered = True
                return {'voice': voice, 'duplicate': False}
        finally:
            if staging.exists(): self._cleanup(staging)

    def edit(self, voice_id, **updates):
        allowed = {'name', 'category', 'usage', 'preferences'}
        if updates.keys() - allowed: raise ValueError('Unsupported voice update.')
        with self.lock():
            doc = self.read()
            voice = next(v for v in doc['voices'] if v['id'] == voice_id)
            voice.update(updates); self.write(doc)

    def save_calibration(self, voice_id, pitch, analysis, reference=None):
        if type(pitch) is not int or not -24 <= pitch <= 24:
            raise ValueError('Invalid calibrated pitch shift.')
        with self.lock():
            doc = self.read()
            voice = next(v for v in doc['voices'] if v['id'] == voice_id)
            if not voice['rvc']['f0'] and pitch:
                raise ValueError('This non-F0 voice supports only zero pitch shift.')
            voice['preferences']['pitch'] = pitch
            voice['calibration'] = {'schema': 1, 'pitch': pitch, 'source_analysis': analysis,
                                    'reference_analysis': reference,
                                    'selected_by': 'user listening choice',
                                    'saved_at': datetime.now(timezone.utc).isoformat()}
            self.write(doc)

    def remove(self, voice_id):
        with self.lock():
            doc = self.read(); voice = next(v for v in doc['voices'] if v['id'] == voice_id)
            doc['voices'].remove(voice); doc['removed'].append(voice)
            if doc['last_voice'] == voice_id: doc['last_voice'] = None
            self.write(doc)

    def delete_files(self, voice_id, confirmed=False):
        if not confirmed: raise ValueError('Explicit confirmation is required to delete imported voice files.')
        with self.lock():
            doc = self.read()
            voice = next(v for v in doc['voices'] + doc['removed'] if v['id'] == voice_id)
            directory = self.path_for(voice, 'model').parent
            if directory.name != voice.get('directory_id', voice['id']) or directory.parent != self.root / 'voices':
                raise ValueError('Refusing to delete files outside this imported voice directory.')
            # Commit removal first. A deletion failure leaves an orphan, never an entry pointing at partial files.
            doc['voices'] = [v for v in doc['voices'] if v['id'] != voice_id]
            doc['removed'] = [v for v in doc['removed'] if v['id'] != voice_id]
            if doc['last_voice'] == voice_id: doc['last_voice'] = None
            self.write(doc)
            self._cleanup(directory)
