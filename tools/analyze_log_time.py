#!/usr/bin/env python3
"""统计 mmdet 训练 log 中每个 epoch 的耗时，以及 time / data_time 的平均值。

用法:
    python tools/analyze_log_time.py <log_file> [<log_file> ...]

对每个 epoch 输出:
    - epoch 墙钟耗时 (由日志时间戳推算)
    - 所有 iter 的 time / data_time 平均值 (按窗口覆盖的 iter 数加权)
    - 去掉每个 epoch 第一条日志 (含 dataloader 预热) 后的 time / data_time 平均值
"""
import argparse
import re
import sys
from datetime import datetime

# 例: 2026-06-21 10:37:44,573 - mmdet - INFO - Epoch [1][10/81]	lr: ..., time: 6.926, data_time: 1.151, ...
LINE_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?'
    r'Epoch \[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?'
    r'time: (?P<time>[\d.]+), data_time: (?P<data_time>[\d.]+)'
)
TS_RE = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
SAVE_RE = re.compile(r'Saving checkpoint at (?P<epoch>\d+) epochs')
START_RE = re.compile(r'workflow: ')

TS_FMT = '%Y-%m-%d %H:%M:%S,%f'


def parse_ts(s):
    return datetime.strptime(s, TS_FMT)


def fmt_dur(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:d}:{m:02d}:{s:02d}'


def analyze(path):
    epochs = {}          # epoch -> list of dict(ts, iter, time, data_time)
    save_ts = {}         # epoch -> timestamp of "Saving checkpoint"
    start_ts = None      # 训练开始时间戳

    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                e = int(m.group('epoch'))
                epochs.setdefault(e, []).append(dict(
                    ts=parse_ts(m.group('ts')),
                    iter=int(m.group('iter')),
                    total=int(m.group('total')),
                    time=float(m.group('time')),
                    data_time=float(m.group('data_time')),
                ))
                continue
            ms = SAVE_RE.search(line)
            if ms:
                tsm = TS_RE.match(line)
                if tsm:
                    save_ts[int(ms.group('epoch'))] = parse_ts(tsm.group('ts'))
                continue
            if start_ts is None and START_RE.search(line):
                tsm = TS_RE.match(line)
                if tsm:
                    start_ts = parse_ts(tsm.group('ts'))

    print(f'==== {path} ====')
    if not epochs:
        print('  未找到 Epoch 日志行')
        return

    grand = dict(t=0.0, d=0.0, n=0, t2=0.0, d2=0.0, n2=0)
    durations = []
    prev_end = start_ts

    for e in sorted(epochs):
        rows = epochs[e]
        # 每条日志覆盖的 iter 数 (当前 iter - 上一条 iter)，用于加权
        weights = []
        prev_iter = 0
        for r in rows:
            weights.append(r['iter'] - prev_iter)
            prev_iter = r['iter']

        def wavg(key, rs, ws):
            tot_w = sum(ws)
            if tot_w == 0:
                return float('nan')
            return sum(r[key] * w for r, w in zip(rs, ws)) / tot_w

        time_all = wavg('time', rows, weights)
        data_all = wavg('data_time', rows, weights)

        # 去掉每个 epoch 的第一条日志 (dataloader 预热)
        rows_skip, weights_skip = rows[1:], weights[1:]
        time_skip = wavg('time', rows_skip, weights_skip) if rows_skip else float('nan')
        data_skip = wavg('data_time', rows_skip, weights_skip) if rows_skip else float('nan')

        n_iter = sum(weights)
        # epoch 墙钟耗时：上一 epoch 结束 (或训练开始) -> 本 epoch 保存 checkpoint
        end = save_ts.get(e, rows[-1]['ts'])
        dur = (end - prev_end).total_seconds() if prev_end else None
        prev_end = end

        dur_str = (f'{fmt_dur(dur)} ({dur/60:.2f} min)'
                   if dur is not None else 'N/A')
        if dur is not None:
            durations.append(dur)
        print(f'Epoch {e:>3}: iters={n_iter:<5} 耗时={dur_str}')
        print(f'          全部     time={time_all:.4f}  data_time={data_all:.4f}')
        print(f'          去首条   time={time_skip:.4f}  data_time={data_skip:.4f}')

        grand['t'] += sum(r['time'] * w for r, w in zip(rows, weights))
        grand['d'] += sum(r['data_time'] * w for r, w in zip(rows, weights))
        grand['n'] += n_iter
        grand['t2'] += sum(r['time'] * w for r, w in zip(rows_skip, weights_skip))
        grand['d2'] += sum(r['data_time'] * w for r, w in zip(rows_skip, weights_skip))
        grand['n2'] += sum(weights_skip)

    print('---- 全部 epoch 汇总 ----')
    if durations:
        avg_dur = sum(durations) / len(durations)
        print(f'  平均每 epoch 耗时 {fmt_dur(avg_dur)} ({avg_dur/60:.2f} min)  '
              f'(epochs={len(durations)})')
    if grand['n']:
        print(f'  全部   time={grand["t"]/grand["n"]:.4f}  '
              f'data_time={grand["d"]/grand["n"]:.4f}  (iters={grand["n"]})')
    if grand['n2']:
        print(f'  去首条 time={grand["t2"]/grand["n2"]:.4f}  '
              f'data_time={grand["d2"]/grand["n2"]:.4f}  (iters={grand["n2"]})')
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='+', help='训练 log 文件路径')
    args = parser.parse_args()
    for path in args.logs:
        try:
            analyze(path)
        except FileNotFoundError:
            print(f'文件不存在: {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
