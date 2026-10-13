"""Check CPU-return completion of the audited control operations, without fitting."""
import argparse
import json
import os
from pathlib import Path

from scripts.measure_afd_control_staging import UUIDS, check_idle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite boundary evidence')
    check_idle()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(UUIDS)
    import torch
    rows = []
    for rank in (0, 1):
        torch.cuda.set_device(rank)
        for batch in (1, 8):
            body = json.dumps(dict(dp_metadata_list={'0': dict(num_tokens_across_dp_cpu=[batch],
                max_tokens_across_dp_cpu=batch)}, is_graph_capturing=False, is_warmup=False,
                is_graph_replaying=False, is_profile=False), separators=(',', ':'), sort_keys=True).encode()
            length = torch.tensor([len(body)], dtype=torch.long, device=rank)
            payload = torch.tensor(list(body), dtype=torch.uint8, device=rank)
            for repeat in range(10):
                torch.cuda.synchronize()
                if rank == 0:
                    cpu = torch.frombuffer(bytearray(body), dtype=torch.uint8)
                    gpu = cpu.to(rank)
                    size = torch.tensor([gpu.numel()], dtype=torch.long, device=rank)
                    complete = torch.cuda.current_stream().query()
                    assert complete and size.item() == len(body) and gpu.cpu().numpy().tobytes() == body
                    rows.append(dict(operation='h2d', batch=batch, repeat=repeat, stream_complete_at_return=complete))
                else:
                    value = int(length.item())
                    complete = torch.cuda.current_stream().query()
                    assert complete and value == len(body)
                    rows.append(dict(operation='length_readback', batch=batch, repeat=repeat, stream_complete_at_return=complete))
                    torch.cuda.synchronize()
                    value = payload.cpu().numpy().tobytes()
                    complete = torch.cuda.current_stream().query()
                    assert complete and value == body
                    rows.append(dict(operation='payload_readback', batch=batch, repeat=repeat, stream_complete_at_return=complete))
    with args.output.open('x') as stream:
        json.dump(dict(scope='boundary_validation_only_not_timing', torch_version=torch.__version__,
                       gpu_uuids=UUIDS, cases=rows, all_passed=True), stream, indent=2)
    print(json.dumps(dict(all_passed=True, cases=len(rows))))


if __name__ == '__main__':
    main()
