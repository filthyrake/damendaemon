"""Attribute worker-side cost separately from socket delivery (not a benchmark)."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import cProfile
import json
import pstats
import time

from damendaemon.scheduler import BatchedRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    runtime = BatchedRuntime({'a': args.path}, context=256)
    messages = [{'role': 'user', 'content': 'Explain how matrix multiplication works in detail, with a worked example.'}]
    def request(_):
        return list(runtime.generate('a', messages, 64))[-1]
    try:
        request(0)
        original = runtime._group
        profiles = []
        def group(*args):
            profile = cProfile.Profile()
            try:
                return profile.runcall(original, *args)
            finally:
                profiles.append(profile)
        runtime._group = group
        with ThreadPoolExecutor(4) as pool:
            start = time.perf_counter()
            results = list(pool.map(request, range(4)))
            print(json.dumps({'seconds': time.perf_counter()-start, 'results': results}, indent=2))
        for profile in profiles:
            pstats.Stats(profile).strip_dirs().sort_stats('cumtime').print_stats(35)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
