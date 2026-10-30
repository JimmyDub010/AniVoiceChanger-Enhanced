# Acknowledgments and bundled SwiftF0 notices

AniVoiceChanger retains its original MIT license and attribution. RVC, PyTorch,
FAISS and separately supplied voice/support models retain their own licenses.
User voice weights, indexes and recordings are never included in release archives.

SwiftF0 0.3.0 by Lars Nieradzik: https://github.com/lars76/swift-f0/tree/v0.3.0
Copyright 2025–2026 Lars Nieradzik. Code and bundled model are distributed under
the repository's MIT license; the wheel includes the model and its license.
The public training repository identifies runs/seed1/model.onnx as the shipped
artifact: https://github.com/lars76/swift-f0-training (also MIT).
The exact original license is retained at
vendor/swiftf0/packages/swift_f0-0.3.0.dist-info/licenses/LICENSE.

ONNX Runtime 1.19.2, copyright Microsoft Corporation, is MIT licensed.
Its LICENSE, Privacy.md and ThirdPartyNotices.txt are retained under
vendor/swiftf0/packages/onnxruntime. CPU extraction uses the official Windows
CPython 3.9 wheel. No vendor installer or modified binary was executed.

The private bundle also retains licenses for coloredlogs 15.0.1 (MIT),
humanfriendly 10.0 (MIT), pyreadline3 3.4.1 (BSD), flatbuffers 23.5.9 (Apache 2.0),
packaging 22.0 (Apache/BSD), protobuf 4.22.1 (BSD), sympy 1.11.1 (BSD), and
mpmath 1.2.1 (BSD). Exact license files and metadata remain in their original
package/dist-info locations. BUNDLE.json records official PyPI download URLs,
wheel SHA-256 values and every bundled file hash. RVC's existing NumPy is used
unchanged (tested 1.23.5); no second NumPy copy is installed.

RMVPE New is the official RVC support checkpoint from the MIT-labeled model
repository lj1995/VoiceConversionWebUI, revision
e6d0c1a17da07c33557852f9dfa2bd44cc75737d. Its SHA-256 matches the tested local
asset. Attribution is retained in release_tools/asset-notices. FCPE's checkpoint
matches Git blob 1741230325796805def4b15aebc12c56c70da974 in CNChTu/FCPE, whose
MIT license is included. FCPE, einops and local-attention licenses are retained
in experiments/fcpe/licenses. local_attention/__init__.py is a minimal compatibility
import shim. PITCH-ASSET-MANIFEST.json records exact hashes and provenance.
These assets use their own licenses, not SwiftF0's. No rights for unrelated voice
models are implied. Personal voices/indexes are never included in release archives.
