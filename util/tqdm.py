import sys
from time import time
from tqdm import tqdm

class TqdmManager:
    def __init__(self, verbose: str, program_id: int):
        self.total = 0
        self.verbose = verbose
        self.program_id = program_id
        self.current = 0
        self.last_pos = 0
        self.start_time = time()
        self.tqdm_bar = None
            
    def set_total(self, total: int):
        self.total = total
        if self.verbose == "tqdm":
            self.tqdm_bar = tqdm(total=self.total)
        elif self.verbose == "simple":
            self.tqdm_bar = 0

    def update(self, req_num: int):
        if self.verbose == "tqdm":
            self.tqdm_bar.update(req_num)
        elif self.verbose == "simple":
            self.tqdm_bar += req_num
            if self.tqdm_bar / self.total + 1e-8 > self.last_pos + 0.005:
                print(
                    f"[{self.program_id}]{self.tqdm_bar} / {self.total} = {self.tqdm_bar / self.total * 100:.2f}% time elapsed {time() - self.start_time}s",
                    file=sys.stderr,
                )
                self.last_pos += 0.005
            elif self.tqdm_bar == self.total:
                print(
                    f"[{self.program_id}]{self.tqdm_bar} / {self.total} = 100.00% time elapsed {time() - self.start_time}s",
                    file=sys.stderr,
                )
