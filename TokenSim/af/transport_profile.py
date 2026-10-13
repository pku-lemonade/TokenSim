"""Scoped direct-message device transport services, excluding CPU and peer wait."""
import math
from TokenSim.af.calibration import content_fingerprint, _plain
from TokenSim.af.snapshot import snapshot, ConfigurationOwner
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import MODEL_VERSION


class TransportServiceProfile(ConfigurationOwner):
    _configuration_fields=('document','fingerprint')
    def __init__(self, document, *, environment, domains, device_indices):
        d=_plain(document)
        if (d.get('schema_version')!=1 or d.get('unit')!='us'
            or d.get('timing_scope')!='transport_service'
            or d.get('contains_wait') is not False or d.get('contains_cpu_submit') is not False
            or d.get('model_version')!=MODEL_VERSION):
            raise ConfigurationError('transport profile must exclude CPU submission and external wait')
        if not environment or d.get('environment')!=_plain(environment):
            raise ConfigurationError('transport profile environment mismatch')
        if d.get('domains')!=_plain(domains) or d.get('device_indices')!=_plain(device_indices):
            raise ConfigurationError('transport profile domain/placement mismatch')
        if not isinstance(d.get('profile_id'),str) or not d['profile_id']:
            raise ConfigurationError('transport profile id required')
        evidence=d.get('evidence',{})
        if (evidence.get('kind') not in ('measured','synthetic_test') or not evidence.get('source')
            or evidence.get('window_count',0)<3 or evidence.get('sample_count',0)<150
            or evidence.get('measurement_boundary')!='after_both_device_kernels_started_until_both_complete'
            or evidence.get('peer_delay_audit_passed') is not True
            or not evidence.get('limitations')):
            raise ConfigurationError('transport boundary and independent delay evidence required')
        rows=d.get('records',[]);keys=set()
        if not rows:raise ConfigurationError('transport profile records required')
        for row in rows:
            key=(row.get('source_device'),row.get('target_device'),row.get('backend'),row.get('dtype'))
            if key in keys or key[0]==key[1] or key[0] not in device_indices.values() or key[1] not in device_indices.values():
                raise ConfigurationError('duplicate or invalid transport endpoints')
            if key[2] not in ('pynccl','torch_distributed_nccl') or key[3] not in ('bf16','int64','uint8'):
                raise ConfigurationError('unsupported measured transport backend/dtype')
            keys.add(key);points=row.get('points',[])
            if not points:raise ConfigurationError('transport measurement points required')
            previous=-1
            for point in points:
                size,value=point.get('bytes'),point.get('service_us')
                if type(size) is not int or size<=previous or size<=0:
                    raise ConfigurationError('transport byte points must increase')
                if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                    raise ConfigurationError('positive finite transport service required')
                previous=size
            if row.get('interpolation')!='linear_within_training_range':
                raise ConfigurationError('explicit bounded interpolation rule required')
        self.document=snapshot(d);self.fingerprint=content_fingerprint(d)

    def estimate_us(self, *, source, target, backend, dtype, size):
        if type(size) is not int or size<0:raise ConfigurationError('invalid transport bytes')
        if size==0:return 0.
        for row in self.document['records']:
            if (source,target,backend,dtype)!=(row['source_device'],row['target_device'],row['backend'],row['dtype']):continue
            points=row['points']
            for point in points:
                if size==point['bytes']:return point['service_us']
            for lo,hi in zip(points,points[1:]):
                if lo['bytes']<size<hi['bytes']:
                    fraction=(size-lo['bytes'])/(hi['bytes']-lo['bytes'])
                    return lo['service_us']+fraction*(hi['service_us']-lo['service_us'])
        raise ConfigurationError('transport profile has no matching in-range message; refusing analytical fallback')
