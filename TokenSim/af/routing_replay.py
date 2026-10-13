"""Indexed native decode assignments; no fallback to synthetic routing."""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict

from TokenSim.af.calibration import content_fingerprint
from TokenSim.af.estimate_cache import freeze
from TokenSim.errors import ConfigurationError
from TokenSim.moe.routing import RoutingAssignment


class _RequestView:
    def __init__(self, request, assignments):
        self._request = request
        self.decode_routing_assignments = assignments

    def __getattr__(self, name):
        return getattr(self._request, name)


class RoutingReplay:
    def __init__(self, document, *, expected_runtime, expected_environment):
        required = {'schema_version', 'runtime_fingerprint', 'environment', 'source', 'kind', 'assignments'}
        if not isinstance(document, Mapping) or set(document) != required:
            raise ConfigurationError('routing replay requires version, runtime/environment, source, kind and assignments')
        if type(document['schema_version']) is not int or document['schema_version'] != 1:
            raise ConfigurationError('unsupported routing replay version')
        if document['runtime_fingerprint'] != expected_runtime:
            raise ConfigurationError('routing replay runtime fingerprint mismatch')
        if not expected_environment or document['environment'] != expected_environment:
            raise ConfigurationError('routing replay environment mismatch or unspecified environment')
        if document['kind'] not in ('captured', 'synthetic_test') or not isinstance(document['source'], str) or not document['source']:
            raise ConfigurationError('routing replay requires explicit captured/synthetic_test provenance')
        if not isinstance(document['assignments'], (list, tuple)) or not document['assignments']:
            raise ConfigurationError('routing replay requires nonempty assignments')
        self.fingerprint = content_fingerprint(document)
        self.kind = document['kind']
        self.source = document['source']
        index = defaultdict(list)
        requests_by_layer = defaultdict(set)
        seen = set()
        for record in document['assignments']:
            if not isinstance(record, Mapping) or 'iteration_id' in record:
                raise ConfigurationError('replay records use decode_step; iteration_id is assigned by the runtime')
            try:
                assignment = RoutingAssignment(iteration_id=0, **record)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f'invalid routing replay record: {exc}') from exc
            if assignment.phase != 'decode' or assignment.a_dp_rank or assignment.a_tp_rank:
                raise ConfigurationError('routing replay supports native 1A1F decode only')
            if assignment.identity in seen:
                raise ConfigurationError('duplicate routing replay assignment')
            seen.add(assignment.identity)
            key = (assignment.decode_step, assignment.layer_id, assignment.request_id)
            requests_by_layer[key[:2]].add(assignment.request_id)
            normalized = asdict(assignment)
            normalized.pop('iteration_id')
            index[key].append(normalized)
        self._index = freeze(index)
        self._requests_by_layer = {key: frozenset(value) for key, value in requests_by_layer.items()}

    def requests_for(self, requests, *, phase, decode_step, layer_id):
        if phase != 'decode':
            raise ConfigurationError('routing replay only supports decode')
        if {r.id for r in requests} != self._requests_by_layer.get((decode_step, layer_id), frozenset()):
            raise ConfigurationError('routing replay request batch does not match the recorded layer')
        result = []
        for request in requests:
            if any(getattr(request, name, None) is not None for name in
                   ('routing_assignments', 'decode_routing_assignments', 'expert_histogram', 'decode_expert_histogram')):
                raise ConfigurationError('routing replay conflicts with request routing override')
            key = (decode_step, layer_id, request.id)
            if key not in self._index:
                raise ConfigurationError(f'missing routing replay records for step/layer/request {key}')
            result.append(_RequestView(request, self._index[key]))
        return result

    def describe(self):
        return dict(kind=self.kind, source=self.source, fingerprint=self.fingerprint,
                    status='assignments_replayed_not_timing_validation')
