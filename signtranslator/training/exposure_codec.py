"""Lossless checkpoint storage for repeated per-example target declarations."""
from copy import deepcopy
import hashlib
import json

from ..reproducibility import canonical_json_bytes
from .target_cells import parse_target_cells


def pack_exposure(records):
    """Intern exact declarations, retaining absent fields and null branches."""
    if not isinstance(records, list):
        raise ValueError('exposure codec requires a record list')
    definitions, packed = {}, []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('exposure codec requires record objects')
        item = deepcopy({key: value for key, value in record.items() if key != 'target_cells'})
        if 'target_cells' in record:
            if not isinstance(record['target_cells'], dict):
                raise ValueError('exposure codec requires target-cell branches')
            branches = {}
            for name, declaration in record['target_cells'].items():
                if declaration is None:
                    branches[name] = None
                    continue
                cells = parse_target_cells(declaration)
                if not cells.examples:
                    raise ValueError('exposure codec requires nonempty example lists')
                references = []
                for row in declaration['examples']:
                    single = {key: value for key, value in declaration.items() if key != 'examples'}
                    single['examples'] = [row]
                    payload = canonical_json_bytes(single)
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest in definitions:
                        if canonical_json_bytes(definitions[digest]) != payload:
                            raise ValueError('exposure declaration digest collision')
                    else:
                        definitions[digest] = json.loads(payload)
                    references.append(digest)
                branches[name] = references
            item['target_cells'] = branches
        packed.append(item)
    return dict(schema_version=1, records=packed, target_cells=definitions)


def unpack_exposure(envelope):
    """Verify the pool and reconstruct independent logical records before load."""
    if (not isinstance(envelope, dict) or set(envelope) != {'schema_version', 'records', 'target_cells'}
            or type(envelope['schema_version']) is not int or envelope['schema_version'] != 1
            or not isinstance(envelope['records'], list) or not isinstance(envelope['target_cells'], dict)):
        raise ValueError('invalid exposure storage envelope')
    definitions = envelope['target_cells']
    for digest, declaration in definitions.items():
        if (type(digest) is not str
                or hashlib.sha256(canonical_json_bytes(declaration)).hexdigest() != digest):
            raise ValueError('exposure declaration digest mismatch')
        if len(parse_target_cells(declaration).examples) != 1:
            raise ValueError('exposure pool must contain single-example declarations')
    used, records = set(), []
    for record in envelope['records']:
        if not isinstance(record, dict):
            raise ValueError('invalid exposure stored record')
        item = deepcopy({key: value for key, value in record.items() if key != 'target_cells'})
        if 'target_cells' in record:
            if not isinstance(record['target_cells'], dict):
                raise ValueError('invalid exposure stored branches')
            branches = {}
            for name, references in record['target_cells'].items():
                if references is None:
                    branches[name] = None
                    continue
                if (not isinstance(references, list) or not references
                        or any(type(digest) is not str or digest not in definitions for digest in references)):
                    raise ValueError('missing or invalid exposure declaration reference')
                base = deepcopy(definitions[references[0]])
                base['examples'] = []
                contract = {key: value for key, value in base.items() if key != 'examples'}
                for digest in references:
                    declaration = definitions[digest]
                    if {key: value for key, value in declaration.items() if key != 'examples'} != contract:
                        raise ValueError('inconsistent exposure declaration reference contracts')
                    base['examples'].append(deepcopy(declaration['examples'][0]))
                    used.add(digest)
                branches[name] = base
            item['target_cells'] = branches
        records.append(item)
    if used != definitions.keys():
        raise ValueError('unused exposure declaration definitions')
    return records
