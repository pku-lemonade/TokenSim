"""Behavior regression of the actual patched guard, without torch/NPU installs."""
import builtins
import types
import unittest
from unittest.mock import patch
from scripts.patch_afd_dbo_cache import OLD_GUARD, patch_source


class DboPatchTests(unittest.TestCase):
    def guard(self):
        source=patch_source('import torch\n\n'+OLD_GUARD)
        ns={}
        importer=builtins.__import__
        def fake_import(name,*a,**kw):
            if name=='torch': return types.SimpleNamespace()
            return importer(name,*a,**kw)
        with patch('builtins.__import__',side_effect=fake_import): exec(source,ns)
        return ns

    def test_missing_npu_is_cached_but_cuda_state_changes(self):
        ns=self.guard(); state=[False]; yielded=[]; imports=[]
        ns.update(dbo_enabled=lambda:state[0],dbo_yield=lambda:yielded.append('cuda'))
        def unavailable(name,*a,**kw):
            imports.append(name); raise ModuleNotFoundError('torch_npu')
        with patch('builtins.__import__',side_effect=unavailable):
            ns['_yield_if_dbo_enabled'](); state[0]=True
            for _ in range(5): ns['_yield_if_dbo_enabled']()
        self.assertEqual(len(imports),1)
        self.assertEqual(yielded,['cuda']*5)

    def test_npu_dynamic_state_and_native_fallback(self):
        ns=self.guard(); state=[False]; yielded=[]; imports=[]
        backend=types.SimpleNamespace(dbo_enabled=lambda:state[0],dbo_yield=lambda:yielded.append('npu'))
        ns.update(dbo_enabled=lambda:True,dbo_yield=lambda:yielded.append('cuda'))
        def available(name,*a,**kw): imports.append(name); return backend
        with patch('builtins.__import__',side_effect=available):
            ns['_yield_if_dbo_enabled'](); state[0]=True
            ns['_yield_if_dbo_enabled'](); state[0]=False
            ns['_yield_if_dbo_enabled']()
        self.assertEqual(yielded,['cuda','npu','cuda'])
        self.assertEqual(len(imports),2)  # two from-imports, one resolution

    def test_non_import_error_propagates(self):
        ns=self.guard()
        with patch('builtins.__import__',side_effect=RuntimeError('broken backend')):
            with self.assertRaisesRegex(RuntimeError,'broken backend'): ns['_yield_if_dbo_enabled']()

    def test_both_disabled_do_not_yield(self):
        ns=self.guard()
        ns.update(dbo_enabled=lambda:False,dbo_yield=lambda:self.fail('unexpected yield'))
        backend=types.SimpleNamespace(dbo_enabled=lambda:False,dbo_yield=lambda:self.fail('unexpected yield'))
        with patch('builtins.__import__',return_value=backend): ns['_yield_if_dbo_enabled']()

    def test_unreviewed_or_already_patched_source_rejected(self):
        with self.assertRaises(ValueError): patch_source('import torch\n'+OLD_GUARD.replace('except ImportError:', 'except Exception:'))
        once=patch_source('import torch\n'+OLD_GUARD)
        with self.assertRaises(ValueError): patch_source(once)


if __name__=='__main__': unittest.main()
