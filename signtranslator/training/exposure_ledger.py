"""Interned resident exposure history with the existing expanded JSON interface."""
import json

from ..reproducibility import canonical_json_bytes
from .exposure_codec import pack_exposure, unpack_exposure


class ExposureLedger:
    """Keep immutable compact records and declarations; expand one record per iteration.

    Resident storage scales with distinct declarations plus record references.
    Consumers that collect the iterator still allocate the full expanded history.
    This class encodes storage, not semantic authorization of optimizer records.
    """

    __slots__ = ('_records', '_definitions')

    def __init__(self, encoded=()):
        self._records = []
        self._definitions = {}
        for record in encoded:
            self.append(record)

    def __len__(self):
        return len(self._records)

    def prepare(self, encoded):
        """Validate/encode before an optimizer call, without changing history."""
        packed = pack_exposure([json.loads(encoded)])
        definitions = tuple((digest, canonical_json_bytes(value))
                            for digest, value in packed['target_cells'].items())
        for digest, payload in definitions:
            previous = self._definitions.get(digest)
            if previous is not None and previous != payload:
                raise ValueError('exposure declaration digest collision')
        return canonical_json_bytes(packed['records'][0]), definitions

    def append_prepared(self, prepared):
        """Commit this ledger's prepared record after the optimizer returns.

        Internal single-writer operation; callers must not forge prepared values.
        """
        record, definitions = prepared
        self._definitions.update(definitions)
        self._records.append(record)

    def append(self, encoded):
        self.append_prepared(self.prepare(encoded))

    def __iter__(self):
        for payload in self._records:
            record = json.loads(payload)
            used = {digest for refs in record.get('target_cells', {}).values()
                    if refs is not None for digest in refs}
            pool = {digest: json.loads(self._definitions[digest]) for digest in used}
            expanded = unpack_exposure(dict(schema_version=1, records=[record], target_cells=pool))
            yield canonical_json_bytes(expanded[0]).decode('utf-8')

    def storage_envelope(self):
        """Copy the pooled schema-5 payload without expanding repeated declarations.

        Semantic history validation remains the trainer's responsibility.
        """
        return dict(schema_version=1,
                    records=[json.loads(payload) for payload in self._records],
                    target_cells={key: json.loads(value) for key, value in self._definitions.items()})

    @property
    def encoded_bytes(self):
        """Stored record/pool payload bytes, excluding Python/container overhead."""
        return sum(map(len, self._records)) + sum(len(key.encode('ascii')) + len(value)
                                                for key, value in self._definitions.items())
