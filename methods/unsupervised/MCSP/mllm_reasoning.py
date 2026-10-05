

import os
import csv
import time
import json
import random
import re
import base64
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: 'openai' package not found. Please install via `pip install openai`")


SampleLike = Union[str, Dict[str, Any], Tuple[str, str]]


class IntentReasoningAgent:
    'Generate intent labels for multimodal clusters from video and text evidence.'


    GEMINI3_PRO_USD_PER_1M_INPUT_TOKENS = 2.0
    GEMINI3_PRO_USD_PER_1M_OUTPUT_TOKENS = 12.0

    def __init__(
        self,
        api_key: str,
        tsv_path: str,
        model_name: str = "qwen3-vl-plus",


        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        verbose: bool = False,
        request_interval: float = 2.0,
        max_retries: int = 5,
        hard_neighbor_margin: float = 0.0,
        dataset: str = "MIntRec",

        max_video_mb: float = 25.0,

        max_text_per_cluster: int = 3,

        video_fps: float = 2.0,

        cost_report_path: Optional[str] = None,
    ):
        self.verbose = bool(verbose)
        self.request_interval = float(request_interval)
        self.max_retries = int(max_retries)

        self.model_name = str(model_name)
        self.base_url = str(base_url)
        self.tsv_path = str(tsv_path)

        self.dataset_raw = dataset or ""
        self.dataset = self.dataset_raw.strip().lower()

        self.max_video_bytes = int(max_video_mb * 1024 * 1024)
        self.max_text_per_cluster = int(max_text_per_cluster)

        self.video_fps = float(video_fps)
        if self.video_fps < 0.1:
            self.video_fps = 0.1
        if self.video_fps > 10.0:
            self.video_fps = 10.0


        self.hard_neighbor_margin = float(hard_neighbor_margin)
        if self.hard_neighbor_margin not in (0.0, 0):
            self._log("INIT", f"注意：hard_neighbor_margin={self.hard_neighbor_margin} 将被忽略；局部对比只使用最近邻簇。")


        self._video_b64_cache: Dict[str, str] = {}


        dataset_root = self.tsv_path
        if os.path.isfile(dataset_root):
            dataset_root = os.path.dirname(dataset_root)
        self.video_root_mintrec = os.environ.get(
            "MCSP_MINTREC_VIDEO_ROOT", os.path.join(dataset_root, "video")
        )
        self.video_root_mintrec2 = os.environ.get(
            "MCSP_MINTREC2_VIDEO_ROOT", os.path.join(dataset_root, "raw_data")
        )
        self.video_root_meldda = os.environ.get(
            "MCSP_MELD_DA_VIDEO_ROOT", os.path.join(dataset_root, "video")
        )


        env_path = os.environ.get("MLLM_COST_REPORT_PATH", "").strip()
        self.cost_report_path = (cost_report_path or env_path or "").strip()
        if not self.cost_report_path:
            self.cost_report_path = os.path.join(os.getcwd(), "mllm_cost_report.json")

        self.cost_stats: Dict[str, Any] = {
            "dataset": self.dataset_raw,
            "dataset_norm": self.dataset,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "tsv_path": self.tsv_path,
            "pricing": {
                "usd_per_1m_input_tokens": self.GEMINI3_PRO_USD_PER_1M_INPUT_TOKENS,
                "usd_per_1m_output_tokens": self.GEMINI3_PRO_USD_PER_1M_OUTPUT_TOKENS,
                "note": "Pricing is hard-coded for rough estimation; update if your billing differs.",
            },
            "usage": {
                "num_calls_text": 0,
                "num_calls_multimodal": 0,
                "num_calls_total": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_prompt_tokens_fallback": 0,
                "estimated_completion_tokens_fallback": 0,
                "video_mb_sent_total": 0.0,
            },
            "cost_estimate_usd": {
                "input_usd": 0.0,
                "output_usd": 0.0,
                "total_usd": 0.0,
            },
            "last_updated_unix": None,
        }


        self.client = None
        if OpenAI is None:
            self._log("INIT", "OpenAI SDK 不可用 -> client=None（无法进行远程推理）")
        elif api_key:
            try:

                self.client = OpenAI(api_key=api_key, base_url=self.base_url)
                self._log("INIT", f"Client 初始化成功 | base_url={self.base_url} | model={self.model_name}")
            except Exception as e:
                self._log("INIT", f"Client 初始化失败 -> {e}")
                self.client = None
        else:
            self._log("INIT", "api_key 为空 -> client=None（无法进行远程推理）")


        self.text_map: Dict[str, str] = self._load_text_data()

        self._log(
            "INIT",
            f"dataset={self.dataset_raw}（norm={self.dataset}）| "
            f"video_roots: MIntRec={self.video_root_mintrec}, "
            f"MIntRec2.0={self.video_root_mintrec2}, "
            f"MELD-DA={self.video_root_meldda}"
        )


    def _log(self, tag: str, msg: str) -> None:
        print(f"[IntentReasoningAgent][{tag}] {msg}")


    @staticmethod
    def _estimate_tokens_fallback(text: str) -> int:
        if not text:
            return 0
        s = re.sub(r"\s+", " ", text.strip())
        return int(np.ceil(len(s) / 4.0))

    def _update_cost_from_tokens(self) -> None:
        u = self.cost_stats["usage"]
        in_tok = int(u["prompt_tokens"])
        out_tok = int(u["completion_tokens"])

        input_usd = (in_tok / 1_000_000.0) * float(self.GEMINI3_PRO_USD_PER_1M_INPUT_TOKENS)
        output_usd = (out_tok / 1_000_000.0) * float(self.GEMINI3_PRO_USD_PER_1M_OUTPUT_TOKENS)

        self.cost_stats["cost_estimate_usd"]["input_usd"] = float(input_usd)
        self.cost_stats["cost_estimate_usd"]["output_usd"] = float(output_usd)
        self.cost_stats["cost_estimate_usd"]["total_usd"] = float(input_usd + output_usd)
        self.cost_stats["last_updated_unix"] = int(time.time())

    def _extract_usage_numbers(self, resp_usage: Any) -> Dict[str, int]:
        out = {
            "billed_prompt": 0,
            "billed_completion": 0,
            "billed_total": 0,
            "completion_reasoning": 0,
            "completion_text": 0,
        }
        if resp_usage is None:
            return out

        pt = getattr(resp_usage, "prompt_tokens", None)
        ct = getattr(resp_usage, "completion_tokens", None)
        tt = getattr(resp_usage, "total_tokens", None)

        out["billed_prompt"] = int(pt) if pt is not None else 0
        out["billed_completion"] = int(ct) if ct is not None else 0
        out["billed_total"] = int(tt) if tt is not None else (out["billed_prompt"] + out["billed_completion"])

        ctd = getattr(resp_usage, "completion_tokens_details", None)
        if ctd is not None:
            rt = getattr(ctd, "reasoning_tokens", None)
            tx = getattr(ctd, "text_tokens", None)
            out["completion_reasoning"] = int(rt) if rt is not None else 0
            out["completion_text"] = int(tx) if tx is not None else 0

        return out

    def _record_usage(
        self,
        *,
        call_type: str,
        prompt_text: str,
        completion_text: str,
        resp_usage: Optional[Any] = None,
        video_mb_sent: float = 0.0,
    ) -> None:
        u = self.cost_stats["usage"]
        if call_type == "text":
            u["num_calls_text"] += 1
        else:
            u["num_calls_multimodal"] += 1
        u["num_calls_total"] += 1

        if video_mb_sent > 0:
            u["video_mb_sent_total"] = float(u["video_mb_sent_total"]) + float(video_mb_sent)

        usage_nums = self._extract_usage_numbers(resp_usage)
        billed_pt = usage_nums["billed_prompt"]
        billed_ct = usage_nums["billed_completion"]
        billed_tt = usage_nums["billed_total"]

        completion_text_tokens = usage_nums["completion_text"]
        completion_reasoning_tokens = usage_nums["completion_reasoning"]

        est_prompt_visible = self._estimate_tokens_fallback(prompt_text)
        est_completion_visible = self._estimate_tokens_fallback(completion_text)

        visible_ct = completion_text_tokens if completion_text_tokens > 0 else est_completion_visible
        visible_pt = est_prompt_visible

        u.setdefault("prompt_tokens_billed", 0)
        u.setdefault("completion_tokens_billed", 0)
        u.setdefault("total_tokens_billed", 0)
        u.setdefault("completion_reasoning_tokens", 0)
        u.setdefault("completion_text_tokens", 0)
        u.setdefault("prompt_tokens_visible_est", 0)
        u.setdefault("completion_tokens_visible_est", 0)

        u["prompt_tokens_billed"] += int(billed_pt)
        u["completion_tokens_billed"] += int(billed_ct)
        u["total_tokens_billed"] += int(billed_tt)

        u["completion_reasoning_tokens"] += int(completion_reasoning_tokens)
        u["completion_text_tokens"] += int(completion_text_tokens)

        u["prompt_tokens_visible_est"] += int(visible_pt)
        u["completion_tokens_visible_est"] += int(visible_ct)

        u["prompt_tokens"] += int(billed_pt)
        u["completion_tokens"] += int(billed_ct)
        u["total_tokens"] += int(billed_tt)

        self._update_cost_from_tokens()

    def save_cost_report(self) -> None:
        try:
            path = self.cost_report_path
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.cost_stats, f, ensure_ascii=False, indent=2)
            self._log("COST", f"已写入成本报告：{path}")
        except Exception as e:
            self._log("COST", f"写入成本报告失败：{e}")


    def _load_text_data(self) -> Dict[str, str]:
        text_map: Dict[str, str] = {}

        root = self.tsv_path
        if not root or not os.path.exists(root):
            self._log("TSV", f"TSV root path not found: {root}")
            return text_map

        if os.path.isfile(root):
            paths = [root]
        else:
            paths = []
            for split in ["train", "dev", "test"]:
                p = os.path.join(root, f"{split}.tsv")
                if os.path.exists(p):
                    paths.append(p)

        total = 0
        for p in paths:
            try:
                self._log("TSV", f"开始读取：{p}")
                with open(p, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    header = next(reader, None)
                    if header is None:
                        continue

                    for line in reader:
                        if not line:
                            continue

                        ds = self.dataset

                        if ds in ("mintrec", "l-mintrec"):
                            if len(line) < 4:
                                continue
                            season = line[0].strip()
                            episode = line[1].strip()
                            clip = line[2].strip()
                            text = line[3].strip()
                            if not (season and episode and clip and text):
                                continue
                            key = f"{season}_{episode}_{clip}"

                        elif ds in ("mintrec2", "mintrec2.0", "mintrec2_0"):
                            if len(line) < 3:
                                continue
                            dialogue_id = line[0].strip()
                            utt_id = line[1].strip()
                            text = line[2].strip()
                            if not (dialogue_id and utt_id and text):
                                continue
                            key = f"dia{dialogue_id}_utt{utt_id}"

                        elif ds in ("meld-da", "meld_da", "meldda"):
                            if len(line) < 3:
                                continue
                            dialogue_id = line[0].strip()
                            utt_id = line[1].strip()
                            text = line[2].strip()
                            if not (dialogue_id and utt_id and text):
                                continue
                            key = f"{dialogue_id}_{utt_id}"

                        elif ds in ("iemocap-da", "iemocap_da"):
                            if len(line) < 2:
                                continue
                            key = line[0].strip()
                            text = line[1].strip()
                            if not (key and text):
                                continue

                        else:
                            if len(line) < 2:
                                continue
                            key = line[0].strip()
                            text = line[1].strip()
                            if not (key and text):
                                continue

                        text_map[key] = text
                        total += 1

                self._log("TSV", f"读取完成：{p} | written={total}")

            except Exception as e:
                self._log("TSV", f"Error loading TSV data from {p}: {e}")

        self._log("TSV", f"text_map 构建完成：size={len(text_map)} | loaded_rows={total}")
        for k in list(text_map.keys())[:3]:
            self._log("TSV", f"sample_key='{k}' | text_len={len(text_map[k])}")
        return text_map


    def build_video_path_from_original_index(self, original_index: str) -> str:
        oi = (original_index or "").strip()
        if not oi:
            return ""

        if self.dataset == "mintrec":
            return os.path.join(self.video_root_mintrec, f"MIntRec_{oi}.mp4")

        if self.dataset in ("mintrec2.0", "mintrec2", "mintrec2_0"):
            return os.path.join(self.video_root_mintrec2, f"{oi}.mp4")

        if self.dataset in ("meld-da", "meld_da", "meldda"):
            parts = oi.split('_')
            filename1 = f"MELD_{parts[0]}_{parts[1]}"
            return os.path.join(self.video_root_meldda, f"{filename1}.mp4")

        return ""


    def _lookup_text(self, original_index: str) -> str:
        oi = (original_index or "").strip()
        if not oi:
            return ""

        if oi in self.text_map:
            return self.text_map[oi]

        if oi.startswith("MIntRec_"):
            cand = oi[len("MIntRec_"):]
            if cand in self.text_map:
                return self.text_map[cand]

        if self.dataset in ("mintrec2.0", "mintrec2", "mintrec2_0"):
            m = re.match(r"^(\d+)[_\-](\d+)$", oi)
            if m:
                cand = f"dia{m.group(1)}_utt{m.group(2)}"
                if cand in self.text_map:
                    return self.text_map[cand]

            m2 = re.match(r"^dia(\d+)_utt(\d+)$", oi)
            if m2:
                cand2 = f"{m2.group(1)}_{m2.group(2)}"
                if cand2 in self.text_map:
                    return self.text_map[cand2]

        return ""


    def _load_representatives_from_csv(
        self,
        rep_csv_path: str,
        max_per_cluster: int = 3,
    ) -> Dict[int, List[Dict[str, str]]]:
        out: Dict[int, List[Dict[str, str]]] = {}

        if not rep_csv_path:
            self._log("CSV", "rep_csv_path 为空")
            return out
        if not os.path.exists(rep_csv_path):
            self._log("CSV", f"rep_csv_path 不存在：{rep_csv_path}")
            return out

        self._log("CSV", f"读取代表样本 CSV：{rep_csv_path}")
        rows = 0

        try:
            with open(rep_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows += 1
                    try:
                        cid = int((r.get("rep_cluster_id") or "").strip())
                    except Exception:
                        continue

                    oi = (r.get("original_index") or "").strip()
                    text = (r.get("text") or "").strip()

                    if (not text) and oi:
                        text = (self._lookup_text(oi) or "").strip()

                    vp = self.build_video_path_from_original_index(oi) if oi else ""
                    item = {"original_index": oi, "text": text, "video_path": vp}

                    if cid not in out:
                        out[cid] = []
                    if len(out[cid]) < max_per_cluster:
                        out[cid].append(item)

            self._log("CSV", f"CSV 读取完成：rows={rows} | clusters={len(out)}")
            for cid in sorted(out.keys())[:5]:
                xs = out[cid]
                ex_oi = xs[0].get("original_index", "")
                ex_txt_len = len(xs[0].get("text", "") or "")
                self._log("CSV", f"cid={cid} samples={len(xs)} | ex_index={ex_oi} | ex_text_len={ex_txt_len}")
            return out

        except Exception as e:
            self._log("CSV", f"CSV 读取失败：{e}")
            return {}


    def generate_intent_definitions_from_reps(
        self,
        rep_texts: Dict[int, List[SampleLike]],
        rep_feats: np.ndarray,
        batch_k: int = 1,
        rep_csv_path: Optional[str] = None,
    ) -> Dict[int, str]:
        self._log("RUN", "===== generate_intent_definitions_from_reps 开始 =====")

        if self.client is None:
            self._log("RUN", "client=None（无法推理）-> 返回空字典")
            return {}

        if rep_feats is None or (not isinstance(rep_feats, np.ndarray)) or rep_feats.ndim != 2:
            self._log("RUN", "rep_feats 非法 -> 返回空字典")
            return {}

        K = int(rep_feats.shape[0])
        self._log("RUN", f"K={K} | batch_k={batch_k} | rep_texts_clusters={len(rep_texts)}")


        csv_path = rep_csv_path or os.environ.get("ICR_REP_CSV", "")
        rep_from_csv = self._load_representatives_from_csv(
            rep_csv_path=csv_path,
            max_per_cluster=self.max_text_per_cluster,
        ) if csv_path else {}

        if rep_from_csv:
            self._log("RUN", f"将使用 CSV 代表样本做多模态推理 | clusters={len(rep_from_csv)}")
        else:
            self._log("RUN", "未提供/未读取到 CSV（ICR_REP_CSV 也为空）-> 将退化为 text-only 推理（不推荐）")


        dist = euclidean_distances(rep_feats, rep_feats)
        np.fill_diagonal(dist, np.inf)
        nn = np.argmin(dist, axis=1).astype(int)


        tasks: List[Dict[str, Any]] = []
        cluster_ids = sorted(rep_from_csv.keys()) if rep_from_csv else sorted(rep_texts.keys())
        for cid in cluster_ids:
            if cid < 0 or cid >= K:
                self._log("RUN", f"警告：cid={cid} 超出范围[0,{K - 1}]，跳过")
                continue
            neighbor = int(nn[cid])
            tasks.append({"cid": int(cid), "neighbor": neighbor, "dist": float(dist[cid, neighbor])})

        self._log("RUN", f"局部对比任务数={len(tasks)}（只用最近邻簇）")
        if self.verbose and tasks:
            self._log("RUN", f"任务示例（前5个）：{tasks[:5]}")


        has_video = bool(rep_from_csv)
        if has_video and int(batch_k) != 1:
            self._log("RUN", f"检测到多模态（视频）推理，强制 batch_k=1（原 batch_k={batch_k}）")
            batch_k = 1

        local_labels = self._local_reasoning(
            rep_texts=rep_texts,
            rep_from_csv=rep_from_csv,
            tasks=tasks,
            batch_k=max(1, int(batch_k)),
        )
        self._log("RUN", f"Local 输出数量={len(local_labels)}")
        if not local_labels:
            self._log("RUN", "Local 为空 -> 返回空字典")
            self.save_cost_report()
            return {}


        refined = self._global_refine(local_labels)
        if refined:
            self._log("RUN", f"Global refine 成功 | 输出数量={len(refined)}")
            self._log("RUN", "===== generate_intent_definitions_from_reps 结束（GLOBAL） =====")
            self.save_cost_report()
            return refined

        self._log("RUN", "Global refine 失败 -> 回退使用 Local 输出")
        self._log("RUN", "===== generate_intent_definitions_from_reps 结束（LOCAL） =====")
        self.save_cost_report()
        return local_labels


    def _local_reasoning(
        self,
        rep_texts: Dict[int, List[SampleLike]],
        rep_from_csv: Dict[int, List[Dict[str, str]]],
        tasks: List[Dict[str, Any]],
        batch_k: int = 1,
    ) -> Dict[int, str]:
        results: Dict[int, str] = {}

        batches: List[List[Dict[str, Any]]] = []
        buf: List[Dict[str, Any]] = []
        for t in tasks:
            buf.append(t)
            if len(buf) >= batch_k:
                batches.append(buf)
                buf = []
        if buf:
            batches.append(buf)

        self._log("LOCAL", f"tasks={len(tasks)} -> batches={len(batches)}（batch_k={batch_k}）")

        for bi, bt in enumerate(batches, start=1):
            cids = [x["cid"] for x in bt]
            self._log("LOCAL", f"---- batch {bi}/{len(batches)} | cids={cids} ----")

            task = bt[0]
            prompt, video_items = self._build_local_prompt_and_videos(rep_texts, rep_from_csv, task)
            raw = self._call_llm_multimodal(prompt_text=prompt, video_items=video_items, context=f"local_batch_{bi}")

            if not raw:
                self._log("LOCAL", f"batch {bi} 返回为空 -> 失败（不会中断训练，继续下一个）")
                continue

            parsed = self._parse_json_labels(raw, key="intent")
            if self.verbose:
                self._log("LOCAL", f"batch {bi} 解析到 keys={sorted(list(parsed.keys()))}")

            for t in bt:
                cid = int(t["cid"])
                if cid in parsed:
                    results[cid] = parsed[cid]
                    self._log("LOCAL", f"cid={cid} | neighbor={t['neighbor']} | dist={t['dist']:.4f} -> '{results[cid]}'")
                else:
                    self._log("LOCAL", f"警告：cid={cid} 不在输出中（可开启 verbose 查看 raw）")

        return results

    def _build_local_prompt_and_videos(
        self,
        rep_texts: Dict[int, List[SampleLike]],
        rep_from_csv: Dict[int, List[Dict[str, str]]],
        task: Dict[str, Any],
        a_max_pairs: int = 3,
        b_max_pairs: int = 3,
    ) -> Tuple[str, List[Dict[str, str]]]:
        cid = int(task["cid"])
        nb = int(task["neighbor"])
        d = float(task["dist"])

        a_max_pairs = max(0, int(a_max_pairs))
        b_max_pairs = max(0, int(b_max_pairs))

        A_csv = rep_from_csv.get(cid, [])
        B_csv = rep_from_csv.get(nb, [])

        def _collect_paired_samples(
            samples: List[Dict[str, str]],
            max_pairs: int,
            group_prefix: str,
        ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
            pairs: List[Dict[str, str]] = []
            video_items: List[Dict[str, str]] = []

            if max_pairs <= 0:
                return pairs, video_items

            for s in samples:
                if len(pairs) >= max_pairs:
                    break

                oi = (s.get("original_index") or "").strip()
                text = (s.get("text") or "").strip()
                vp = (s.get("video_path") or "").strip()

                if not text or not vp:
                    continue

                b64 = self._read_video_b64(vp)
                if not b64:
                    continue

                tag = f"{group_prefix}{len(pairs) + 1}"
                pairs.append(
                    {
                        "tag": tag,
                        "text": text,
                        "original_index": oi,
                        "video_path": vp,
                    }
                )
                video_items.append(
                    {
                        "name": f"{tag}_{oi}",
                        "path": vp,
                        "b64": b64,
                    }
                )

            return pairs, video_items

        A_pairs, A_videos = _collect_paired_samples(A_csv, max_pairs=a_max_pairs, group_prefix="A")
        B_pairs, B_videos = _collect_paired_samples(B_csv, max_pairs=b_max_pairs, group_prefix="B")

        if not A_pairs:
            A_texts: List[str] = []
            if A_csv:
                for s in A_csv[: max(1, self.max_text_per_cluster)]:
                    t = (s.get("text") or "").strip()
                    if t:
                        A_texts.append(t)
            else:
                for s in (rep_texts.get(cid, []) or [])[: max(1, self.max_text_per_cluster)]:
                    if isinstance(s, str) and s.strip():
                        A_texts.append(s.strip())

            take_n = a_max_pairs if a_max_pairs > 0 else min(3, len(A_texts))
            A_pairs = [
                {"tag": f"A{i+1}", "text": t, "original_index": "", "video_path": ""}
                for i, t in enumerate(A_texts[:take_n])
            ]

        if (not B_pairs) and (b_max_pairs > 0):
            B_texts: List[str] = []
            if B_csv:
                for s in B_csv:
                    t = (s.get("text") or "").strip()
                    if t:
                        B_texts.append(t)
                    if len(B_texts) >= b_max_pairs:
                        break
            else:
                xs = rep_texts.get(nb, []) or []
                for s in xs:
                    if isinstance(s, str) and s.strip():
                        B_texts.append(s.strip())
                    if len(B_texts) >= b_max_pairs:
                        break

            if B_texts:
                B_pairs = [
                    {"tag": f"B{i+1}", "text": t, "original_index": "", "video_path": ""}
                    for i, t in enumerate(B_texts[:b_max_pairs])
                ]

        video_items: List[Dict[str, str]] = []
        video_items.extend(A_videos)
        video_items.extend(B_videos)

        def _tag_from_video_item_name(name: str) -> str:
            if not name:
                return ""
            return name.split("_", 1)[0].strip()

        video_order_tags = [_tag_from_video_item_name(it.get("name", "")) for it in video_items]
        video_order_tags = [t for t in video_order_tags if t]
        video_order_str = ", ".join(video_order_tags) if video_order_tags else "(no videos)"

        A_block = "\n".join([f"- [{p['tag']}] {p['text']}" for p in A_pairs]) if A_pairs else "- (empty)"
        B_block = "\n".join([f"- [{p['tag']}] {p['text']}" for p in B_pairs]) if B_pairs else "- (empty)"

        prompt = (
            "You are an expert in UNSUPERVISED INTENT DISCOVERY for multimodal dialogues.\n"
            "You will be given short videos and their spoken text (the words spoken in the video) for one FOCUS CLUSTER (Group A) and one NEAREST NEIGHBOR CLUSTER (Group B).\n\n"
            "========================\n"
            "GOAL\n"
            "========================\n"
            "Extract ONE ABSTRACT communicative-function label for Group A ONLY.\n"
            "Group B is a contrast set (hard negative) provided ONLY to improve the accuracy and discriminativeness of the label for Group A.\n\n"
            "========================\n"
            "WHAT 'COMMUNICATIVE FUNCTION' MEANS\n"
            "========================\n"
            "It is the speaker's pragmatic goal / speech act:\n"
            "- what they are trying to achieve with their words,\n"
            "- how they want to affect or influence the listener.\n"
            "It should be at the level of dialogue act / communicative function,\n"
            "NOT at the level of topic, story, or concrete situation.\n\n"
            "========================\n"
            "EVIDENCE & ALIGNMENT (VIDEO + SPOKEN TEXT)\n"
            "========================\n"
            f"- The videos are provided in this exact order: {video_order_str}.\n"
            "- Match each video to the spoken text with the same tag (A1/A2/A3/B1).\n"
            "- Some spoken text may not have a corresponding video; use spoken-text-only in that case.\n"
            "- If video and spoken text conflict, prioritize the video.\n\n"
            "========================\n"
            "YOU MUST NOT\n"
            "========================\n"
            "- Do NOT describe the scenario, location, or concrete event.\n"
            "- Do NOT summarize the story.\n"
            "- Do NOT mention specific people, places, products, or named entities.\n"
            "- Do NOT describe topic content.\n"
            "- Do NOT mix multiple different functions in one label.\n"
            "- Do NOT write a complete sentence.\n"
            "- Do NOT mention dataset label names, label codes, or taxonomies.\n\n"
            "========================\n"
            "STYLE REQUIREMENTS\n"
            "========================\n"
            "- Output a SHORT PHRASE (2–6 English words).\n"
            "- Use a verb phrase or gerund-style phrase.\n"
            "- The phrase must be GENERIC and REUSABLE across many scenarios.\n"
            "- Avoid pure emotion words unless clearly used as a communicative act.\n\n"
            "========================\n"
            "YOUR TASK (DISCRIMINATIVE CONTRAST)\n"
            "========================\n"
            "1) Infer the dominant shared communicative function in Group A (ignore outliers if A1/A2/A3 differ).\n"
            "2) Compare against Group B and REMOVE any label that is also clearly applicable to B (too generic).\n"
            "3) Output ONE fine-grained, highly discriminative label for A that best separates A from B.\n"
            "   If two labels fit A, choose the one LESS applicable to B.\n"
            "   Only output the same label for A and B if they are genuinely indistinguishable.\n\n"
            "========================\n"
            "OUTPUT FORMAT (STRICT)\n"
            "========================\n"
            "Return a JSON array with exactly one object:\n"
            f"[{{\"cid\": {cid}, \"intent\": \"<2-6 word phrase>\"}}]\n"
            "Make sure the output is VALID JSON.\n"
            "Do NOT include any commentary outside the JSON.\n\n"
            f"Task: cid={cid} (Group A) vs neighbor={nb} (Group B) | dist={d:.4f}\n\n"
            "Group A spoken text:\n"
            f"{A_block}\n\n"
            "Group B spoken text (contrast set):\n"
            f"{B_block}\n"
        )

        self._log(
            "LOCAL",
            f"构造 prompt 完成 | cid={cid} | "
            f"A_pairs={len(A_pairs)} | B_pairs={len(B_pairs)} | "
            f"videos={len(video_items)} (A_videos={len(A_videos)}, B_videos={len(B_videos)}) | "
            f"a_max_pairs={a_max_pairs}, b_max_pairs={b_max_pairs}"
        )
        return prompt, video_items


    def _global_refine(self, local_labels: Dict[int, str]) -> Dict[int, str]:
        self._log("GLOBAL", "===== 全局归一开始（text-only） =====")

        items = [{"cid": int(cid), "intent": str(lbl)} for cid, lbl in sorted(local_labels.items())]
        prompt = (
            "You are given cluster labels produced by local reasoning.\n"
            "Your job is FORMAT NORMALIZATION ONLY.\n"
            "This is NOT semantic merging, NOT re-labeling, and NOT re-clustering.\n\n"
            "What you MUST do (format-only edits):\n"
            "- lowercase the label\n"
            "- remove punctuation and quotes\n"
            "- remove articles (a/an/the) and extra filler words\n"
            "- enforce 2–6 English words\n"
            "- enforce a verb phrase or gerund-style phrase.\n\n"
            "What you MUST NOT do:\n"
            "- Do NOT change the underlying meaning of any label.\n"
            "- Do NOT replace words with synonyms.\n"
            "- Do NOT generalize or broaden labels.\n"
            "- Do NOT make two different cids share the same label unless they were already EXACTLY identical in the input.\n"
            "- Do NOT add new semantic content.\n\n"
            "Output constraints:\n"
            "- Return STRICT JSON as an array.\n"
            "- The output MUST contain exactly one object per input item, with the same cids.\n"
            "- Do not add or drop any cid.\n\n"
            f"Input JSON:\n{json.dumps(items, ensure_ascii=False)}\n\n"
            "Return JSON:\n"
            "[{\"cid\": <int>, \"refined_intent\": \"<normalized label>\"}, ...]\n"
            "No extra text.\n"
        )

        raw = self._call_llm_text(prompt, context="global_refine")
        if not raw:
            self._log("GLOBAL", "全局归一返回为空 -> 失败")
            return {}

        parsed = self._parse_json_labels(raw, key="refined_intent")
        if not parsed:
            self._log("GLOBAL", "全局归一 JSON 解析失败 -> 失败")
            if self.verbose:
                self._log("GLOBAL", f"raw:\n{raw}")
            return {}

        refined: Dict[int, str] = {}
        for cid, lbl in local_labels.items():
            refined[cid] = parsed.get(cid, lbl)

        for cid in sorted(refined.keys())[:10]:
            self._log("GLOBAL", f"cid={cid}: '{local_labels[cid]}' -> '{refined[cid]}'")

        self._log("GLOBAL", "===== 全局归一成功 =====")
        return refined


    def _call_llm_text(self, prompt: str, context: str) -> str:
        if self.client is None:
            self._log("CALL", f"{context} | client=None -> skip")
            return ""

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            time.sleep(self._sleep_time(attempt))
            try:
                if self.verbose:
                    self._log("CALL", f"{context} | attempt {attempt}/{self.max_retries} -> send(text)")
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = (resp.choices[0].message.content or "").strip()
                if not raw:
                    raise RuntimeError("Empty response content")

                try:
                    self._record_usage(
                        call_type="text",
                        prompt_text=prompt,
                        completion_text=raw,
                        resp_usage=getattr(resp, "usage", None),
                        video_mb_sent=0.0,
                    )
                except Exception as e:
                    self._log("COST", f"{context} | record_usage(text) failed: {e}")

                if self.verbose:
                    self._log("CALL", f"{context} | raw:\n{raw}")
                return raw
            except Exception as e:
                last_err = e
                if self._is_retryable(e):
                    self._log("CALL", f"{context} | 临时错误：{e} -> 将重试")
                    continue
                self._log("CALL", f"{context} | 非重试错误：{e}")
                break

        self._log("CALL", f"{context} | 重试失败，last_err={last_err}")
        return ""


    def _call_llm_multimodal(self, prompt_text: str, video_items: List[Dict[str, str]], context: str) -> str:
        if self.client is None:
            self._log("CALL", f"{context} | client=None -> skip")
            return ""

        if not video_items:
            self._log("CALL", f"{context} | 无视频可发送 -> 改为 text-only")
            return self._call_llm_text(prompt_text, context=context + "_text_fallback")

        b64_list = [it["b64"] for it in video_items if it.get("b64")]
        self._log("CALL", f"{context} | 准备发送视频数量={len(b64_list)}")
        for it in video_items:
            self._log("CALL", f"{context} | video_item name={it.get('name','')} | path={it.get('path','')} | b64_len={len(it.get('b64',''))}")


        video_mb_sent = 0.0
        try:
            total_b64_chars = sum(len(x) for x in b64_list)
            video_mb_sent = float(total_b64_chars) / (1024.0 * 1024.0)
        except Exception:
            video_mb_sent = 0.0


        messages = self._build_messages_qwen_video_url(prompt_text=prompt_text, b64_list=b64_list)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            time.sleep(self._sleep_time(attempt))
            try:
                if self.verbose:
                    self._log("CALL", f"{context} | qwen_video_url | attempt {attempt}/{self.max_retries} -> send(multimodal)")
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if not raw:
                    raise RuntimeError("Empty response content")

                try:
                    self._record_usage(
                        call_type="multimodal",
                        prompt_text=prompt_text,
                        completion_text=raw,
                        resp_usage=getattr(resp, "usage", None),
                        video_mb_sent=video_mb_sent,
                    )
                except Exception as e:
                    self._log("COST", f"{context} | record_usage(multimodal) failed: {e}")

                if self.verbose:
                    self._log("CALL", f"{context} | raw:\n{raw}")
                self._log("CALL", f"{context} | qwen_video_url | 成功")
                return raw
            except Exception as e:
                last_err = e
                if self._is_retryable(e):
                    self._log("CALL", f"{context} | qwen_video_url | 临时错误：{e} -> 将重试")
                    continue
                self._log("CALL", f"{context} | qwen_video_url | 非重试错误（可能该模型/后端不支持视频文件输入）：{e}")
                break

        self._log("CALL", f"{context} | 多模态失败，last_err={last_err} -> fallback text-only")
        return self._call_llm_text(prompt_text, context=context + "_text_fallback_after_mm_fail")

    def _build_messages_qwen_video_url(self, prompt_text: str, b64_list: List[str]) -> List[Dict[str, Any]]:
        'Build OpenAI-compatible video and text message content.'
        parts: List[Dict[str, Any]] = []

        for b64 in b64_list:

            data_url = f"data:video/mp4;base64,{b64}"
            parts.append(
                {
                    "type": "video_url",
                    "video_url": {
                        "url": data_url,
                        "fps": self.video_fps,
                    },
                }
            )

        parts.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": parts}]


    def _read_video_b64(self, path: str) -> str:
        if not path:
            return ""
        if path in self._video_b64_cache:
            if self.verbose:
                self._log("VIDEO", f"命中缓存：{path}")
            return self._video_b64_cache[path]

        if not os.path.exists(path):
            self._log("VIDEO", f"未找到视频：{path}")
            return ""

        try:
            bs = open(path, "rb").read()
            size = len(bs)
            self._log("VIDEO", f"读取视频成功：{size / 1024 / 1024:.2f} MB | {path}")

            if size > self.max_video_bytes:
                self._log(
                    "VIDEO",
                    f"警告：视频大小 {size / 1024 / 1024:.2f} MB 超过阈值 {self.max_video_bytes / 1024 / 1024:.2f} MB，可能导致接口失败"
                )

            b64 = base64.b64encode(bs).decode("utf-8")
            self._video_b64_cache[path] = b64
            return b64
        except Exception as e:
            self._log("VIDEO", f"读取失败：{path} | err={e}")
            return ""


    @staticmethod
    def _strip_code_fences(s: str) -> str:
        s = (s or "").strip()
        m = re.match(r"^```[a-zA-Z]*\s*([\s\S]*?)```$", s)
        return m.group(1).strip() if m else s

    @staticmethod
    def _postprocess_label(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        s = s.replace("**", "").replace("`", "").replace('"', "")
        s = s.rstrip(",").strip()
        if s.endswith("."):
            s = s[:-1].strip()
        if len(s) > 80:
            s = s[:77] + "..."
        return s

    def _parse_json_labels(self, raw: str, key: str = "intent") -> Dict[int, str]:
        text = self._strip_code_fences(raw)

        try:
            data = json.loads(text)
            out: Dict[int, str] = {}
            if isinstance(data, list):
                for obj in data:
                    if not isinstance(obj, dict):
                        continue
                    if "cid" not in obj:
                        continue
                    try:
                        cid = int(obj["cid"])
                    except Exception:
                        continue

                    val = obj.get(key) or obj.get("intent") or obj.get("refined_intent") or obj.get("label")
                    if isinstance(val, str) and val.strip():
                        out[cid] = self._postprocess_label(val)
            if out:
                return out
        except Exception:
            pass

        out2: Dict[int, str] = {}
        pattern = re.compile(
            r'"cid"\s*:\s*(\d+)[\s\S]*?"(?:intent|refined_intent|label)"\s*:\s*"([^"]+)"',
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            cid = int(m.group(1))
            lab = self._postprocess_label(m.group(2))
            if lab:
                out2[cid] = lab
        return out2


    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        es = str(e).lower()
        return ("503" in es) or ("unavailable" in es) or ("timeout" in es) or ("temporarily" in es) or ("rate limit" in es)

    def _sleep_time(self, attempt: int) -> float:
        if attempt <= 1:
            return self.request_interval
        backoff = (2 ** (attempt - 2)) + random.uniform(0, 1)
        return self.request_interval + backoff
