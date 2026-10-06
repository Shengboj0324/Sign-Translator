"""Run from the repository root; fictional-corpus checkpoint boundary probe."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_trainability_contract import make
from test_supported_trainer import mixed

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    vocabulary, corpus = mixed(root)
    source = make(corpus, vocabulary, frozen=True)
    path = source.save(root / 'frozen.pt')
    receiver = make(corpus, vocabulary, frozen=False)
    try:
        receiver.load(path)
        refusal = None
    except ValueError as failure:
        refusal = str(failure)
    print(json.dumps(dict(source_requires_grad=source.model.byte_embedding.weight.requires_grad,
                          receiver_requires_grad=receiver.model.byte_embedding.weight.requires_grad,
                          resume_refusal=refusal, receiver_steps=receiver.global_step,
                          phase_exit_approved=False), indent=2))
