#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_logger.py — 训练过程结构化日志工具

控制方式:
  REWARD_DEBUG=0  每批汇总 (默认)
  REWARD_DEBUG=1  逐样本详细
  REWARD_DEBUG=2  +JSONL 文件到 REWARD_LOG_DIR
"""

import os
import sys
import json
import time
from collections import defaultdict
from pathlib import Path

_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text

def _red(t):    return _c(t, "31")
def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _cyan(t):   return _c(t, "36")
def _bold(t):   return _c(t, "1")

def _debug_level() -> int:
    try:
        return int(os.environ.get("REWARD_DEBUG", "0"))
    except ValueError:
        return 0

def _log_dir() -> Path:
    d = Path(os.environ.get("REWARD_LOG_DIR", "/tmp/reward_logs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class RewardLogger:
    def __init__(self, role: str):
        self.role = role
        self.level = _debug_level()
        self._batch_records = []
        self._sample_count = 0
        self._reward_sum = 0.0
        self._reward_buckets = defaultdict(int)
        self._start_time = time.time()

        self._jsonl_path = None
        if self.level >= 2:
            ts = int(time.time())
            self._jsonl_path = _log_dir() / f"{role}_reward_{ts}.jsonl"
            print(f"  [RewardLogger] 日志文件: {self._jsonl_path}")

    def log_challenger_sample(
        self, generated, category, gate, label_verified,
        adv_success, topic_sim, reward, signal_source,
    ):
        self._sample_count += 1
        self._reward_sum += reward
        bucket = self._reward_bucket(reward)
        self._reward_buckets[bucket] += 1

        record = {
            "role": "challenger", "n": self._sample_count,
            "category": category,
            "generated_preview": generated[:80].replace("\n", "↵"),
            "gate": round(gate, 3),
            "label_verified": label_verified,
            "adversarial_success": adv_success,
            "topic_sim": round(topic_sim, 3),
            "reward": round(reward, 4),
            "signal": signal_source,
        }
        self._batch_records.append(record)

        if self.level >= 1:
            self._print_challenger_sample(record, generated)
        if self.level >= 2 and self._jsonl_path:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_reviewer_sample(
        self, text_preview, pred, true_cat, true_is_harmful,
        base_score, bonus, reward,
    ):
        self._sample_count += 1
        self._reward_sum += reward
        bucket = self._reward_bucket(reward)
        self._reward_buckets[bucket] += 1

        pred_cat = pred.get("category", "?")
        pred_bin = pred.get("binary", "?")
        true_bin = "有害" if true_is_harmful else "无害"

        record = {
            "role": "reviewer", "n": self._sample_count,
            "text_preview": text_preview[:60].replace("\n", "↵"),
            "true_category": true_cat, "true_binary": true_bin,
            "pred_category": pred_cat, "pred_binary": pred_bin,
            "cat_ok": (pred_cat == true_cat), "bin_ok": (pred_bin == true_bin),
            "base_score": round(base_score, 3),
            "bonus": round(bonus, 3),
            "reward": round(reward, 4),
        }
        self._batch_records.append(record)

        if self.level >= 1:
            self._print_reviewer_sample(record)
        if self.level >= 2 and self._jsonl_path:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_batch_summary(self):
        if not self._batch_records:
            return
        n = len(self._batch_records)
        avg_r = self._reward_sum / self._sample_count if self._sample_count else 0
        batch_avg = sum(r["reward"] for r in self._batch_records) / n
        elapsed = time.time() - self._start_time

        pos = sum(1 for r in self._batch_records if r["reward"] > 0)
        neg = sum(1 for r in self._batch_records if r["reward"] < 0)
        zer = n - pos - neg

        header = _bold(f"  ── [{self.role.upper()} REWARD SUMMARY] ──")
        print(f"\n{header}")
        print(f"  本批样本数: {n}   全局已处理: {self._sample_count}")
        print(f"  本批平均奖励: {_fmt_reward(batch_avg)}   全局平均: {_fmt_reward(avg_r)}")
        print(f"  奖励分布: {_green(f'+{pos}')} 正  {_red(f'-{neg}')} 负  {_yellow(f'~{zer}')} 零附近")

        if self.role == "challenger":
            sig_counts = defaultdict(int)
            for r in self._batch_records:
                sig_counts[r.get("signal", "?")] += 1
            parts = "  ".join(f"{k}:{v}" for k, v in sig_counts.items())
            print(f"  信号来源: {_cyan(parts)}")
            gates = [r["gate"] for r in self._batch_records]
            print(f"  quality gate: avg={sum(gates)/len(gates):.3f}  min={min(gates):.3f}  max={max(gates):.3f}")

        if self.role == "reviewer":
            bin_ok = sum(1 for r in self._batch_records if r.get("bin_ok"))
            cat_ok = sum(1 for r in self._batch_records if r.get("cat_ok"))
            print(f"  二分类准确率: {bin_ok}/{n} ({bin_ok/n:.0%})")
            print(f"  类别准确率:   {cat_ok}/{n} ({cat_ok/n:.0%})")

        print(f"  累计耗时: {elapsed:.1f}s\n")
        self._batch_records.clear()

    @staticmethod
    def _reward_bucket(r: float) -> str:
        if r >= 0.5:  return "+++"
        if r >= 0.0:  return "+  "
        if r >= -0.3: return "-  "
        return "---"

    def _print_challenger_sample(self, record, full_text):
        n, reward, sig = record["n"], record["reward"], record["signal"]
        sig_tag = {"phase_a": _green("[Phase A]"), "partial": _yellow("[Partial]"),
                   "fallback": _red("[Fallback]")}.get(sig, sig)
        print(f"  #{n:04d} {sig_tag} cat={record['category']:<8} gate={record['gate']:.2f} "
              f"lv={_bool_icon(record['label_verified'])} adv={_bool_icon(record['adversarial_success'])} "
              f"→ R={_fmt_reward(reward)}")
        print(f"         生成: {_cyan(full_text.strip()[:100].replace(chr(10), '↵'))}")

    def _print_reviewer_sample(self, record):
        n, reward = record["n"], record["reward"]
        bin_icon = _green("✓") if record["bin_ok"] else _red("✗")
        cat_icon = _green("✓") if record["cat_ok"] else _red("✗")
        print(f"  #{n:04d} [{record['true_binary']}→{record['pred_binary']}]{bin_icon} "
              f"[{record['true_category']}→{record['pred_category']}]{cat_icon} "
              f"base={record['base_score']:.2f} bonus={record['bonus']:+.2f} → R={_fmt_reward(reward)}")
        print(f"         文本: {_cyan(record['text_preview'])}")


def _fmt_reward(r: float) -> str:
    if r >= 0.3:   return _green(f"{r:+.4f}")
    if r >= -0.1:  return _yellow(f"{r:+.4f}")
    return _red(f"{r:+.4f}")

def _bool_icon(v) -> str:
    if v is True:  return _green("T")
    if v is False: return _red("F")
    return _yellow("?")
