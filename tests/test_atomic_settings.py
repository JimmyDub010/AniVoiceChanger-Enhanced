import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voice_library


class AtomicSettingsTests(unittest.TestCase):
    def test_transient_windows_lock_recovers_without_losing_existing_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            target=Path(directory)/'settings.json'
            voice_library.atomic_json(target, {'pitch': 0})
            original=voice_library.os.replace
            calls=[]
            def replace(source, destination):
                calls.append(source)
                if len(calls)==1:
                    self.assertEqual(json.loads(target.read_text()), {'pitch': 0})
                    raise PermissionError('Simulated Windows sharing lock')
                return original(source,destination)
            with patch.object(voice_library.os,'replace',side_effect=replace), patch.object(voice_library.time,'sleep'):
                voice_library.atomic_json(target, {'pitch': 12})
            self.assertEqual(len(calls),2)
            self.assertEqual(json.loads(target.read_text()), {'pitch': 12})
            self.assertEqual(list(Path(directory).glob('*.tmp-*')),[])

    def test_permanent_lock_preserves_previous_settings_and_reports_failure(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            target=Path(directory)/'settings.json'
            voice_library.atomic_json(target, {'pitch': -12})
            with patch.object(voice_library.os,'replace',side_effect=PermissionError('Permanent lock')) as replace, patch.object(voice_library.time,'sleep'):
                with self.assertRaises(PermissionError): voice_library.atomic_json(target, {'pitch': 12})
            self.assertEqual(replace.call_count,4)
            self.assertEqual(json.loads(target.read_text()), {'pitch': -12})
            self.assertEqual(list(Path(directory).glob('*.tmp-*')),[])
