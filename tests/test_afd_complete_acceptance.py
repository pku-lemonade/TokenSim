import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_afd_complete_acceptance import frozen_profile_sha


class FrozenEvidenceTest(unittest.TestCase):
    def test_original_bytes_and_semantic_content_are_both_required(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'outputs') as tmp:
            root=Path(tmp);path=root/'docs/plans/profile.json'
            path.parent.mkdir(parents=True)
            document={'fixed_us':4.5,'scope':'host_submission_service'}
            raw=json.dumps(document,indent=2).replace('\n','\r\n').encode()
            path.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
            bundle={'host_profile':document,'implementation_sha256':{'docs/plans/profile.json':digest}}
            self.assertEqual(frozen_profile_sha(bundle,root,'profile.json','host_profile'),digest)
            path.write_bytes(raw.replace(b'\r\n',b'\n'))
            with self.assertRaises(ValueError):
                frozen_profile_sha(bundle,root,'profile.json','host_profile')
            path.write_bytes(raw)
            bundle['host_profile']={'fixed_us':5.0}
            with self.assertRaises(ValueError):
                frozen_profile_sha(bundle,root,'profile.json','host_profile')


if __name__=='__main__':unittest.main()
