import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from global_hotkeys import GlobalKeyBridge


class HookCleanupTests(unittest.TestCase):
    def test_one_registration_dispatches_both_transitions_and_cleans_once(self):
        import keyboard
        controller=Mock()
        bridge=GlobalKeyBridge(controller,lambda function,*args:function(*args))
        callback=[]
        remover=Mock()
        def register(key,function,suppress=False):
            callback.append(function); return remover
        with patch.object(keyboard,'hook_key',side_effect=register) as hook, patch.object(keyboard,'unhook') as unhook:
            bridge.enable('f8')
            callback[0](SimpleNamespace(event_type='down'))
            callback[0](SimpleNamespace(event_type='up'))
            self.assertEqual([c.args for c in controller.event.call_args_list],[('down',),('up',)])
            bridge.disable(); bridge.disable()
            hook.assert_called_once()
            unhook.assert_called_once_with(remover)
