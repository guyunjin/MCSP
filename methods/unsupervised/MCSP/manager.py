

import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import time
import csv
from typing import Dict, List, Tuple, Any

from transformers import BertTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import trange, tqdm

from losses import loss_map
from utils.functions import save_model, restore_model, set_torch_seed
from backbones.base import freeze_bert_parameters
from utils.metrics import clustering_score
from .pretrain import PretrainMCSPManager
from data.utils import get_dataloader
from .utils import set_optimizer, get_pseudo_dataloader

from .mllm_reasoning import IntentReasoningAgent
from .intent_concepts_loader import load_intent_concepts

from utils.metrics import hungray_aligment
from utils.visual import visualize_epoch_clustering_withid

class MCSPManager:
    'Coordinate MCSP training, concept anchoring, and graph label propagation.'

    def __init__(self, args, data, model):
        set_torch_seed(args.seed)
        self.args = args


        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model.model


        mm = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.test_dataloader = mm["train"], mm["test"]
        self.train_outputs = data.train_outputs


        self.tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_bert_model, do_lower_case=True
        )


        self.supcon = loss_map["SupConLoss"]
        self.unsupcon = loss_map["SupConLoss"]


        self.num_labels = args.num_labels
        self.max_rounds = args.num_train_epochs
        self.grad_clip = args.grad_clip


        self.use_unsup_branch = getattr(args, "use_unsup_branch", False)


        self.q_weight_D = getattr(args, "quality_weight_D", 0.0)
        self.q_weight_M = getattr(args, "quality_weight_M", 1.0)


        self.centroids: np.ndarray = None
        self.hq_ids: set = set()
        self.uq_ids: set = set()


        self.intent_anchors: np.ndarray = None
        self.intent_texts: Dict[int, str] = {}


        self.seed_gids_per_concept: Dict[int, List[int]] = {}


        self.concept_tau = getattr(args, "concept_tau", 0.1)
        self.lp_edge_lambda = getattr(args, "lp_edge_lambda", 1.0)
        self.lp_conf_top_ratio = getattr(args, "lp_conf_top_ratio", 0.4)


        self.use_intents_as_seeds = getattr(args, "use_intents_as_seeds", False)
        self.logger.info(f"use_intents_as_seeds = {self.use_intents_as_seeds}")


        self.use_llm = getattr(args, "use_llm", False)


        self.intent_concepts_path = getattr(
            args,
            "intent_concepts_path",
            os.path.join(os.path.dirname(__file__), "intent_concepts.json"),
        )


        raw_api_key = getattr(args, "api_key", "") or os.environ.get("MCSP_API_KEY", "")

        if self.use_llm and raw_api_key:
            api_key_for_agent = raw_api_key
            self.logger.info(
                "use_llm=True and gemini_api_key provided: LLM reasoning ENABLED."
            )
        else:
            api_key_for_agent = ""
            if not raw_api_key:
                self.logger.warning(
                    "No 'gemini_api_key' in args: LLM reasoning DISABLED, "
                    "but TSV text_map will still be loaded for debug / CSV export."
                )
            elif not self.use_llm:
                self.logger.info(
                    "use_llm=False: LLM reasoning DISABLED, "
                    "but TSV text_map will still be loaded for debug / CSV export."
                )

        self.reasoning_agent = IntentReasoningAgent(
            api_key=api_key_for_agent,
            tsv_path=getattr(args, "tsv_path", "") or os.path.join(args.data_path, args.dataset),
            model_name=getattr(args, "llm_model_name", "qwen3-max"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            verbose=getattr(args, "llm_verbose", False),
            request_interval=getattr(args, "llm_request_interval", 5.0),
            max_retries=getattr(args, "llm_max_retries", 5),
            hard_neighbor_margin=getattr(args, "llm_neighbor_margin", 0.05),
            dataset=args.dataset,
        )


        pretrain_manager = PretrainMCSPManager(args, data, model)
        if args.pretrain:

            self.pretrained_model = pretrain_manager.model
            self._load_pretrained_backbone(self.pretrained_model)
        else:

            self.pretrained_model = restore_model(
                pretrain_manager.model,
                os.path.join(args.model_output_path, "pretrain"),
                self.device,
            )
            self._load_pretrained_backbone(self.pretrained_model)


        if args.train:
            self.optimizer, self.scheduler = set_optimizer(args, self.model, args.lr)
            if getattr(args, "freeze_train_bert_parameters", False):
                self.logger.info(
                    "Freeze backbone parameters except final heads for efficiency"
                )
                self.model = freeze_bert_parameters(
                    self.model, args.multimodal_method
                )
        else:
            self.model = restore_model(
                self.model, args.model_output_path, self.device
            )


        os.makedirs(self.args.model_output_path, exist_ok=True)
        self.metrics_file = os.path.join(
            self.args.model_output_path, "training_metrics_icr_id.csv"
        )
        with open(self.metrics_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "sched_ratio",
                    "hq_size",
                    "uq_size",
                    "sup_loss",
                    "unsup_loss",
                    "ACC",
                    "NMI",
                    "ARI",
                    "FMI",
                    "epoch_total_s",
                    "graph_rw_total_s",
                    "random_walk_s",
                    "refresh_seed_s",
                    "knn_s",
                    "pconcept_s",
                    "modulate_s",
                    "seedmat_s",
                    "select_bookkeep_s",
                    "train_s",
                    "extract_s",
                    "kmeans_eval_s",
                ]
            )

        self.timing_file = os.path.join(
            self.args.model_output_path, "training_time_breakdown_icr_id.csv"
        )
        self.timing_summary_file = os.path.join(
            self.args.model_output_path, "training_time_summary_icr_id.csv"
        )
        self.timing_totals: Dict[str, float] = {}
        self.timing_counts: Dict[str, int] = {}
        with open(self.timing_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "epoch", "seconds", "extra"])


    def _record_time(
        self,
        stage: str,
        seconds: float,
        epoch: Any = "",
        extra: str = "",
    ):
        seconds = float(seconds)
        self.timing_totals[stage] = self.timing_totals.get(stage, 0.0) + seconds
        self.timing_counts[stage] = self.timing_counts.get(stage, 0) + 1

        self.logger.info(
            f"[Time][{stage}] epoch={epoch} seconds={seconds:.4f}"
            + (f" | {extra}" if extra else "")
        )

        with open(self.timing_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([stage, epoch, f"{seconds:.6f}", extra])

    def _write_timing_summary(self):
        with open(
            self.timing_summary_file, "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "total_seconds", "count", "avg_seconds"])
            for stage in sorted(self.timing_totals):
                total = self.timing_totals[stage]
                count = self.timing_counts[stage]
                writer.writerow(
                    [stage, f"{total:.6f}", count, f"{total / count:.6f}"]
                )

        self.logger.info(
            f"[Time] Timing summary written to {self.timing_summary_file}"
        )


    def _embed_views(self, text, video, audio):
        'Create normalized multimodal views for contrastive learning.'
        _, out_ta = self.model(
            text,
            torch.zeros_like(video).to(self.device),
            audio,
            mode="train-mm",
        )
        _, out_tv = self.model(
            text,
            video,
            torch.zeros_like(audio).to(self.device),
            mode="train-mm",
        )
        _, out_tva = self.model(text, video, audio, mode="train-mm")

        out_ta = F.normalize(out_ta, dim=-1)
        out_tv = F.normalize(out_tv, dim=-1)
        out_tva = F.normalize(out_tva, dim=-1)

        return torch.stack([out_ta, out_tv, out_tva], dim=1)

    def _extract_feats(self, which: str) -> Dict[str, np.ndarray]:
        'Extract fused features, labels, and sample indices.'
        dl = self.train_dataloader if which == "train" else self.test_dataloader
        self.model.eval()
        feats, labels, idxs = [], [], []

        with torch.no_grad():
            running = 0
            for batch in tqdm(dl, desc=f"Embed ({which})"):
                text = batch["text_feats"].to(self.device)
                video = batch["video_feats"].to(self.device)
                audio = batch["audio_feats"].to(self.device)
                label = batch["label_ids"].to(self.device)

                f, _ = self.model(text, video, audio, mode="train-mm")

                feats.append(f.cpu())
                labels.append(label.cpu())

                bs = f.size(0)
                idxs.append(torch.arange(running, running + bs))
                running += bs

        feats = torch.cat(feats, 0).numpy()
        y_true = torch.cat(labels, 0).numpy()
        idx = torch.cat(idxs, 0).numpy()

        return {"feats": feats, "y_true": y_true, "idx": idx}


    def _cluster(self, feats: np.ndarray) -> np.ndarray:
        'Cluster features with warm-started KMeans when available.'
        if self.centroids is None:
            km = KMeans(
                n_clusters=self.num_labels,
                init="k-means++",
                n_init=10,
                random_state=self.args.seed,
            ).fit(feats)
        else:
            km = KMeans(
                n_clusters=self.num_labels,
                init=self.centroids,
                n_init=1,
                random_state=self.args.seed,
            ).fit(feats)

        self.centroids = km.cluster_centers_
        return km.labels_


    def _compute_quality_scores(self, feats: np.ndarray, assign: np.ndarray) -> np.ndarray:
        'Score samples using cluster density and separation margins.'

        n, d = feats.shape
        K = self.num_labels


        centroids = np.zeros((K, d), dtype=np.float32)
        for cid in range(K):
            idxs = np.where(assign == cid)[0]
            if len(idxs) == 0:
                continue
            centroids[cid] = feats[idxs].mean(axis=0)
        self.centroids = centroids


        dist_all = euclidean_distances(feats, centroids)
        rows = np.arange(n)
        own_dist = dist_all[rows, assign]
        dist_all[rows, assign] = np.inf
        nearest_other = dist_all.min(axis=1)
        M = nearest_other - own_dist


        D = np.zeros(n, dtype=np.float32)
        k = getattr(self.args, "quality_density_k", 5)

        for cid in range(K):
            idxs = np.where(assign == cid)[0]
            if len(idxs) <= 1:
                continue

            cluster_feats = feats[idxs]
            n_c = cluster_feats.shape[0]
            k_eff = min(k, n_c - 1)

            dist_mat = euclidean_distances(cluster_feats, cluster_feats)
            sorted_dists = np.sort(dist_mat, axis=1)[:, 1:1 + k_eff]
            mean_knn = sorted_dists.mean(axis=1)
            density_score = -mean_knn

            D[idxs] = density_score.astype(np.float32)


        def normalize(x: np.ndarray) -> np.ndarray:
            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                return (x - x_min) / (x_max - x_min)
            else:
                return np.zeros_like(x, dtype=np.float32)

        D_norm = normalize(D)
        M_norm = normalize(M.astype(np.float32))


        wD = self.q_weight_D
        wM = self.q_weight_M
        if wD + wM <= 1e-8:
            wD, wM = 0.5, 0.5
        else:
            s = wD + wM
            wD, wM = wD / s, wM / s

        Q = wD * D_norm + wM * M_norm

        self.logger.info(
            f"[Quality] D_norm mean={D_norm.mean():.4f}, "
            f"M_norm mean={M_norm.mean():.4f}, Q mean={Q.mean():.4f}, "
            f"wD={wD:.2f}, wM={wM:.2f}"
        )
        return Q

    def _select_hq_by_quality(
        self,
        quality_scores: np.ndarray,
        idx: np.ndarray,
        sched_ratio: float,
    ):
        'Select the highest-quality samples for warm-up training.'
        n = len(idx)
        k = max(1, int(n * sched_ratio))
        order = np.argsort(-quality_scores)
        top_positions = order[:k]
        top_gids = idx[top_positions]

        self.hq_ids = set(int(g) for g in top_gids.tolist())
        self.uq_ids = set(int(g) for g in idx.tolist()) - self.hq_ids

        self.logger.info(
            f"HQ Selection (Quality): ratio={sched_ratio:.4f}, "
            f"HQ Size={len(self.hq_ids)}, UQ Size={len(self.uq_ids)}"
        )


    def _generate_intent_anchors(self, intent_texts: Dict[int, str]) -> np.ndarray:
        'Encode intent phrases as normalized multimodal anchors.'
        self.model.eval()
        feat_dim = getattr(self.args, "feat_dim", 768)
        anchors = np.zeros((self.num_labels, feat_dim), dtype=np.float32)

        text_list: List[str] = []
        valid_cids: List[int] = []
        for cid in range(self.num_labels):
            if cid in intent_texts:
                text_list.append(intent_texts[cid])
                valid_cids.append(cid)

        if not text_list:
            self.logger.warning("No intent texts provided, anchors will be all zeros.")
            return anchors

        with torch.no_grad():

            encoded = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            if "token_type_ids" in encoded:
                token_type_ids = encoded["token_type_ids"]
            else:
                token_type_ids = torch.zeros_like(input_ids)

            text_feats = torch.stack(
                [input_ids, attention_mask, token_type_ids], dim=1
            ).to(self.device)
            bs = text_feats.size(0)


            try:
                ref_batch = next(iter(self.train_dataloader))
            except StopIteration:
                self.logger.warning(
                    "train_dataloader is empty when building intent anchors; anchors will remain zeros."
                )
                return anchors

            ref_video = ref_batch["video_feats"].to(self.device)
            ref_audio = ref_batch["audio_feats"].to(self.device)
            video_shape = ref_video.shape[1:]
            audio_shape = ref_audio.shape[1:]

            dummy_video = torch.zeros(
                (bs, *video_shape),
                device=self.device,
                dtype=ref_video.dtype,
            )
            dummy_audio = torch.zeros(
                (bs, *audio_shape),
                device=self.device,
                dtype=ref_audio.dtype,
            )


            feats, _ = self.model(text_feats, dummy_video, dummy_audio, mode="train-mm")
            feats = F.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)

            for i, cid in enumerate(valid_cids):
                anchors[cid] = feats[i]

        return anchors

    def _build_intent_anchors_once(
        self,
        feats: np.ndarray,
        idx: np.ndarray,
        assign: np.ndarray,
        quality_scores: np.ndarray,
    ):
        'Build representative samples, intent anchors, and graph seeds.'
        if self.reasoning_agent is None and self.use_llm:
            self.logger.warning(
                "Reasoning agent is None while use_llm=True, skip _build_intent_anchors_once."
            )
            return

        self.logger.info(
            "=== Building intent anchors once (per-cluster top-k & rep-cluster top-3) ==="
        )
        t_build_total = time.perf_counter()

        n = feats.shape[0]
        if n == 0:
            self.logger.warning(
                "[RepSelect] Empty feats when building intent anchors, skip."
            )
            return

        top_k = getattr(self.args, "rep_top_k_per_cluster", 10)
        candidate_positions: List[int] = []
        t0 = time.perf_counter()


        for cid in range(self.num_labels):
            cluster_positions = np.where(assign == cid)[0]
            if len(cluster_positions) == 0:
                self.logger.warning(
                    f"[RepSelect] No samples in original cluster {cid} when selecting candidates."
                )
                continue

            cluster_q = quality_scores[cluster_positions]
            k_eff = min(top_k, len(cluster_positions))
            order_local = np.argsort(-cluster_q)[:k_eff]
            chosen_pos = cluster_positions[order_local]

            candidate_positions.extend(chosen_pos.tolist())

        if len(candidate_positions) == 0:
            self.logger.warning(
                "[RepSelect] No candidate positions collected, skip building anchors."
            )
            return

        candidate_positions = sorted(list(set(candidate_positions)))
        feats_cand = feats[candidate_positions]
        idx_cand = idx[candidate_positions]
        quality_cand = quality_scores[candidate_positions]

        self.logger.info(
            f"[RepSelect] Per original-cluster top-{top_k} candidates collected: "
            f"N_candidates={len(candidate_positions)}, K={self.num_labels}"
        )
        self._record_time(
            "repselect_candidate_collection",
            time.perf_counter() - t0,
            extra=(
                f"N={n}, K={self.num_labels}, top_k={top_k}, "
                f"candidates={len(candidate_positions)}"
            ),
        )


        if len(feats_cand) < self.num_labels:
            self.logger.warning(
                f"[RepSelect] #candidates ({len(feats_cand)}) < num_labels ({self.num_labels}), "
                f"KMeans may degenerate."
            )

        t0 = time.perf_counter()
        km = KMeans(
            n_clusters=self.num_labels,
            init="k-means++",
            n_init=10,
            random_state=self.args.seed,
        ).fit(feats_cand)
        rep_assign = km.labels_
        rep_centers = km.cluster_centers_
        self._record_time(
            "repselect_candidate_kmeans",
            time.perf_counter() - t0,
            extra=f"candidates={len(candidate_positions)}, K={self.num_labels}",
        )


        t0 = time.perf_counter()
        rep_num_examples = getattr(self.args, "rep_num_examples", 3)

        rep_global_ids: Dict[int, List[int]] = {}
        rep_texts: Dict[int, List[str]] = {}

        all_original_indexes = self.train_outputs["indexes"]
        all_labels = self.train_outputs["label_ids"]

        for cid in range(self.num_labels):
            locs = np.where(rep_assign == cid)[0]
            if len(locs) == 0:
                self.logger.warning(f"[RepSelect] No samples in representative cluster {cid}.")
                continue

            cluster_q = quality_cand[locs]
            order_local = np.argsort(-cluster_q)
            m = min(rep_num_examples, len(locs))
            chosen_rel = order_local[:m]
            chosen_pos_rel = locs[chosen_rel]

            gid_list: List[int] = []
            text_list: List[str] = []

            for rel in chosen_pos_rel:
                pos_global = candidate_positions[rel]
                gid = int(idx[pos_global])
                gid_list.append(gid)


                text = ""
                if 0 <= gid < len(all_original_indexes):
                    orig_id = all_original_indexes[gid]
                    if self.reasoning_agent and self.reasoning_agent.text_map:
                        text = (
                            self.reasoning_agent.text_map.get(str(orig_id))
                            or self.reasoning_agent.text_map.get(orig_id)
                            or ""
                        )
                text_list.append(text if text else f"[GID {gid}]")

            rep_global_ids[cid] = gid_list
            rep_texts[cid] = text_list


        out_csv = os.path.join(self.args.model_output_path, "icr_id_representatives.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["rep_cluster_id", "global_id", "original_index", "quality", "label_id", "text"]
            )
            for cid, gid_list in rep_global_ids.items():
                for gid in gid_list:
                    if 0 <= gid < len(all_original_indexes):
                        orig_id = all_original_indexes[gid]
                    else:
                        orig_id = ""

                    pos_arr = np.where(idx == gid)[0]
                    q_val = float(quality_scores[pos_arr[0]]) if len(pos_arr) > 0 else 0.0

                    if 0 <= gid < len(all_labels):
                        label_id = int(all_labels[gid])
                    else:
                        label_id = -1

                    text = ""
                    if 0 <= gid < len(all_original_indexes) and self.reasoning_agent:
                        if self.reasoning_agent.text_map:
                            text = (
                                self.reasoning_agent.text_map.get(str(orig_id))
                                or self.reasoning_agent.text_map.get(orig_id)
                                or ""
                            )

                    writer.writerow([cid, gid, orig_id, q_val, label_id, text])

        self.logger.info(f"[RepSelect] Exported representative samples to {out_csv}")
        self._record_time(
            "repselect_examples_and_export",
            time.perf_counter() - t0,
            extra=f"rep_num_examples={rep_num_examples}, csv={out_csv}",
        )


        t0 = time.perf_counter()
        self.logger.info(
            "[RepSelect] Getting intent phrases for representative clusters..."
        )

        if self.use_llm:
            if self.reasoning_agent is None:
                raise RuntimeError(
                    "use_llm=True, but reasoning_agent is not initialized."
                )

            start_time = time.perf_counter()

            intent_texts = (
                self.reasoning_agent.generate_intent_definitions_from_reps(
                    rep_texts=rep_texts,
                    rep_feats=rep_centers,
                    batch_k=getattr(self.args, "llm_batch_clusters", 1),
                    rep_csv_path=out_csv,
                )
            )

            reasoning_time = time.perf_counter() - start_time

            self.logger.info(
                "[RepSelect] MLLM reasoning time: %.4f seconds",
                reasoning_time,
            )
        else:
            intent_texts = load_intent_concepts(
                concepts_path=self.intent_concepts_path,
                dataset=self.args.dataset,
                seed=self.args.seed,
                num_labels=self.num_labels,
            )

            self.logger.info(
                "[RepSelect] Loaded %d intent concepts from %s "
                "(dataset=%s, seed=%s)",
                len(intent_texts),
                self.intent_concepts_path,
                self.args.dataset,
                self.args.seed,
            )

        self._record_time(
            "mllm_or_fallback_concept_generation",
            time.perf_counter() - t0,
            extra=(
                f"use_llm={self.use_llm}, "
                f"reps_per_cluster={rep_num_examples}"
            ),
        )

        self.intent_texts = intent_texts

        self.logger.info("[Intent List]: %s", self.intent_texts)
        self.logger.info(
            "[RepSelect] Got %d intent phrases.",
            len(self.intent_texts),
        )


        t0 = time.perf_counter()
        self.intent_anchors = self._generate_intent_anchors(self.intent_texts)
        self.logger.info("[RepSelect] Intent anchors have been built and cached.")
        self._record_time(
            "anchor_encoding",
            time.perf_counter() - t0,
            extra=f"num_concepts={len(self.intent_texts)}",
        )


        t0 = time.perf_counter()
        if getattr(self, "use_intents_as_seeds", False):

            gid2pos = {int(g): i for i, g in enumerate(idx)}
            seed_gids_per_concept: Dict[int, List[int]] = {}

            for cid in range(self.num_labels):
                gid_list = rep_global_ids.get(cid, [])
                if not gid_list:
                    self.logger.warning(
                        f"[LabelProp] No representative sample for concept {cid} "
                        f"when use_intents_as_seeds=True; skip this concept."
                    )
                    continue

                gid = int(gid_list[0])
                seed_gids_per_concept[cid] = [gid]

                if cid < self.intent_anchors.shape[0] and gid in gid2pos:
                    pos = gid2pos[gid]
                    feats[pos] = self.intent_anchors[cid].astype(feats.dtype)
                else:
                    self.logger.warning(
                        f"[LabelProp] Cannot assign intent anchor to gid={gid}, cid={cid}: "
                        f"cid or gid not found."
                    )

            self.seed_gids_per_concept = seed_gids_per_concept
            self.logger.info(
                "[LabelProp] use_intents_as_seeds=True: "
                "each concept uses 1 seed sample whose feature is replaced by its intent anchor. "
                f"Seeds: { {cid: gids for cid, gids in self.seed_gids_per_concept.items()} }"
            )
        else:

            self.seed_gids_per_concept = rep_global_ids.copy()
            self.logger.info(
                f"[LabelProp] Seed gids per concept prepared (sample-based): "
                f"{ {cid: len(gids) for cid, gids in self.seed_gids_per_concept.items()} }"
            )
        self._record_time(
            "seed_preparation",
            time.perf_counter() - t0,
            extra=(
                f"use_intents_as_seeds={self.use_intents_as_seeds}, "
                f"num_seed_concepts={len(self.seed_gids_per_concept)}"
            ),
        )
        self._record_time(
            "build_intent_anchors_total",
            time.perf_counter() - t_build_total,
            extra=(
                f"K={self.num_labels}, reps_per_cluster={rep_num_examples}, "
                f"use_llm={self.use_llm}"
            ),
        )


    def _build_seed_label_matrix_and_mask(
        self,
        N: int,
        gidx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        'Build one-hot graph seeds and their mask.'
        Y = np.zeros((N, self.num_labels), dtype=np.float32)
        seed_mask = np.zeros(N, dtype=bool)

        if not self.seed_gids_per_concept:
            self.logger.warning(
                "[LabelProp] seed_gids_per_concept is empty, label propagation will degenerate."
            )
            return Y, seed_mask


        gid2pos = {int(g): i for i, g in enumerate(gidx)}

        for cid, gid_list in self.seed_gids_per_concept.items():
            for gid in gid_list:
                gid = int(gid)
                if gid not in gid2pos:
                    continue
                pos = gid2pos[gid]
                Y[pos, cid] = 1.0
                seed_mask[pos] = True

        num_seeds = seed_mask.sum()
        self.logger.info(
            f"[LabelProp] Built seed label matrix: N={N}, K={self.num_labels}, #seeds={num_seeds}"
        )
        return Y, seed_mask

    def _build_knn_graph(
        self,
        feats: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        'Construct a batched cosine-similarity k-nearest-neighbor graph.'
        feats = feats.astype(np.float32)
        N, D = feats.shape

        k = getattr(self.args, "lp_knn_k", 10)
        tau_g = getattr(self.args, "lp_tau_g", 0.1)
        batch_size = getattr(self.args, "lp_batch_size", 256)


        norm = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats_norm = feats / norm

        neighbors = np.zeros((N, k), dtype=np.int64)
        weights = np.zeros((N, k), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start

            batch = feats_norm[start:end]


            sims = np.dot(batch, feats_norm.T)


            for i in range(B):
                global_i = start + i
                row = sims[i]


                row[global_i] = -1e9

                if k < N:
                    topk_idx = np.argpartition(row, -k)[-k:]
                else:
                    topk_idx = np.arange(N)

                topk_sim = row[topk_idx]


                w = np.exp(topk_sim / tau_g).astype(np.float32)
                w_sum = float(w.sum()) + 1e-8
                w = w / w_sum

                neighbors[global_i] = topk_idx
                weights[global_i] = w

        self.logger.info(
            f"[LabelProp] kNN graph built (manual, batched): "
            f"N={N}, k={k}, batch_size={batch_size}"
        )
        return neighbors, weights

    def _label_propagation(
        self,
        neighbors: np.ndarray,
        weights: np.ndarray,
        Y: np.ndarray,
        seed_mask: np.ndarray,
    ) -> np.ndarray:
        'Propagate seed labels across the weighted k-nearest-neighbor graph.'

        N, K = Y.shape
        Q = Y.astype(np.float32).copy()

        alpha = getattr(self.args, "lp_alpha", 0.9)
        T = getattr(self.args, "lp_steps", 50)

        for t in range(T):

            Q_neighbors = Q[neighbors]
            propagated = (weights[:, :, None] * Q_neighbors).sum(axis=1)

            Q = alpha * propagated + (1.0 - alpha) * Y


            if seed_mask is not None and seed_mask.any():
                Q[seed_mask] = Y[seed_mask]


        Q = Q - Q.max(axis=1, keepdims=True)
        Q = np.exp(Q)
        Q_sum = Q.sum(axis=1, keepdims=True) + 1e-8
        Q = Q / Q_sum

        return Q


    def _compute_concept_distribution_from_anchors(
        self,
        feats: np.ndarray,
    ) -> np.ndarray:
        'Compute sample distributions over intent anchors.'
        N = feats.shape[0]
        K = self.num_labels

        if self.intent_anchors is None:
            self.logger.warning(
                "[Concept] intent_anchors is None, fallback to uniform concept distribution."
            )
            return np.full((N, K), 1.0 / K, dtype=np.float32)

        X = feats.astype(np.float32)
        C = self.intent_anchors.astype(np.float32)


        x_norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        c_norm = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
        Xn = X / x_norm
        Cn = C / c_norm


        sims = np.dot(Xn, Cn.T)

        tau_c = max(self.concept_tau, 1e-6)
        sims = sims / tau_c

        sims = sims - sims.max(axis=1, keepdims=True)
        exp_sims = np.exp(sims).astype(np.float32)
        denom = exp_sims.sum(axis=1, keepdims=True) + 1e-8
        P = exp_sims / denom

        return P

    def _modulate_edge_weights_with_concepts(
        self,
        neighbors: np.ndarray,
        weights_geom: np.ndarray,
        P_concept: np.ndarray,
    ) -> np.ndarray:
        'Reweight graph edges using concept-distribution similarity.'
        N, k = neighbors.shape
        weights_geom = weights_geom.astype(np.float32)
        weights_new = np.zeros_like(weights_geom, dtype=np.float32)

        lambda_edge = self.lp_edge_lambda
        if lambda_edge <= 0.0:

            return weights_geom

        for i in range(N):
            nbr_idx = neighbors[i]
            p_i = P_concept[i]
            p_nbr = P_concept[nbr_idx]


            sim_sem = (p_nbr * p_i[None, :]).sum(axis=1)

            factor = 1.0 + lambda_edge * sim_sem
            w = weights_geom[i] * factor

            s = float(w.sum())
            if s <= 1e-8:
                w = np.full_like(w, 1.0 / k, dtype=np.float32)
            else:
                w = w / s

            weights_new[i] = w

        self.logger.info(
            "[Concept] Edge weights modulated by concept similarity "
            f"(lambda_edge={lambda_edge:.3f})."
        )
        return weights_new


    def _pre_round_build_intent_anchors(self, feats, gidx, y_true):
        'Run warm-up training and initialize intent anchors and seeds.'
        if self.reasoning_agent is None and self.use_llm:
            self.logger.warning(
                "[PreRound] No reasoning agent while use_llm=True; skip semantic anchor pre-round."
            )
            return feats, gidx, y_true

        self.logger.info(
            "=== PreRound: build intent anchors before epoch 0 (treat as epoch -1) ==="
        )
        t_preround_total = time.perf_counter()


        t0 = time.perf_counter()
        assign0 = self._cluster(feats)
        self._record_time(
            "preround_initial_kmeans", time.perf_counter() - t0, epoch=-1
        )

        t0 = time.perf_counter()
        quality0 = self._compute_quality_scores(feats, assign0)
        self._record_time(
            "preround_initial_quality", time.perf_counter() - t0, epoch=-1
        )


        top_k = getattr(self.args, "rep_top_k_per_cluster", 10)
        candidate_positions = []
        t0 = time.perf_counter()

        for cid in range(self.num_labels):
            cluster_pos = np.where(assign0 == cid)[0]
            if len(cluster_pos) == 0:
                self.logger.warning(
                    f"[PreRound] Cluster {cid} has no samples when selecting candidates."
                )
                continue

            cluster_q = quality0[cluster_pos]
            k_eff = min(top_k, len(cluster_pos))
            order_local = np.argsort(-cluster_q)[:k_eff]
            chosen = cluster_pos[order_local]
            candidate_positions.extend(chosen.tolist())

        if len(candidate_positions) == 0:
            self.logger.warning(
                "[PreRound] No candidates collected; skip anchor building."
            )
            return feats, gidx, y_true

        candidate_positions = sorted(list(set(candidate_positions)))
        candidate_gids = gidx[candidate_positions]

        self.logger.info(
            f"[PreRound] Collected candidates from per-cluster top-{top_k}: "
            f"N_candidates={len(candidate_positions)}"
        )
        self._record_time(
            "preround_candidate_selection",
            time.perf_counter() - t0,
            epoch=-1,
            extra=f"top_k={top_k}, candidates={len(candidate_positions)}",
        )


        self.hq_ids = set(int(g) for g in candidate_gids.tolist())
        self.uq_ids = set()

        pseudo_labels0 = assign0.copy()
        self.logger.info(
            f"[PreRound] Start warm-up training on candidates (epoch = -1), "
            f"HQ size={len(self.hq_ids)}"
        )
        t0 = time.perf_counter()
        _ = self._train_one_epoch(pseudo_labels=pseudo_labels0, epoch=-1)
        self._record_time(
            "preround_warmup_train",
            time.perf_counter() - t0,
            epoch=-1,
            extra=f"HQ={len(self.hq_ids)}",
        )


        t0 = time.perf_counter()
        pack_new = self._extract_feats("train")
        self._record_time(
            "preround_reextract_features", time.perf_counter() - t0, epoch=-1
        )
        feats_new, gidx_new, y_true_new = (
            pack_new["feats"],
            pack_new["idx"],
            pack_new["y_true"],
        )

        t0 = time.perf_counter()
        assign1 = self._cluster(feats_new)
        self._record_time(
            "preround_refined_kmeans", time.perf_counter() - t0, epoch=-1
        )

        t0 = time.perf_counter()
        quality1 = self._compute_quality_scores(feats_new, assign1)
        self._record_time(
            "preround_refined_quality", time.perf_counter() - t0, epoch=-1
        )


        t0 = time.perf_counter()
        self._build_intent_anchors_once(
            feats=feats_new,
            idx=gidx_new,
            assign=assign1,
            quality_scores=quality1,
        )
        self._record_time(
            "preround_build_anchors_call",
            time.perf_counter() - t0,
            epoch=-1,
        )
        self.logger.info("[PreRound] Semantic intent anchors & concept seeds ready.")
        self._record_time(
            "preround_total", time.perf_counter() - t_preround_total, epoch=-1
        )

        return feats_new, gidx_new, y_true_new


    def _train_one_epoch(self, pseudo_labels: np.ndarray, epoch: int) -> Tuple[float, float]:
        'Train one warm-up epoch on quality-selected samples.'
        self.model.train()


        tr_sup, steps_sup = 0.0, 0
        if len(self.hq_ids) > 0:
            hq_list = sorted(list(self.hq_ids))
            self.train_outputs["label_ids"] = pseudo_labels
            self.train_outputs["select_ids"] = hq_list

            _, sup_loader = get_pseudo_dataloader(
                args=self.args, train_outputs=self.train_outputs, mode="pretrain"
            )

            for batch in tqdm(sup_loader, desc=f"Train Sup (HQ, Ep={epoch})"):
                text = batch["text_feats"].to(self.device)
                video = batch["video_feats"].to(self.device)
                audio = batch["audio_feats"].to(self.device)
                labels = batch["label_ids"].to(self.device)

                views = self._embed_views(text, video, audio)

                loss = self.supcon(
                    views,
                    labels=labels,
                    temperature=self.args.train_temperature_sup,
                    device=self.device,
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                tr_sup += loss.item()
                steps_sup += 1

        avg_sup_loss = tr_sup / max(1, steps_sup)


        tr_unsup, steps_unsup = 0.0, 0
        if self.use_unsup_branch and len(self.uq_ids) > 0:
            uq_list = sorted(list(self.uq_ids))
            self.train_outputs["select_ids"] = uq_list
            _, unsup_loader = get_pseudo_dataloader(
                args=self.args, train_outputs=self.train_outputs, mode="pretrain"
            )

            for batch in tqdm(unsup_loader, desc=f"Train Unsup (UQ, Ep={epoch})"):
                text = batch["text_feats"].to(self.device)
                video = batch["video_feats"].to(self.device)
                audio = batch["audio_feats"].to(self.device)

                views = self._embed_views(text, video, audio)

                loss = self.unsupcon(
                    views,
                    temperature=self.args.train_temperature_unsup,
                    device=self.device,
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                tr_unsup += loss.item()
                steps_unsup += 1

        avg_unsup_loss = tr_unsup / max(1, steps_unsup)
        self.logger.info(
            f"[Epoch {epoch}] Loss Sup: {avg_sup_loss:.4f}, "
            f"Loss Unsup: {avg_unsup_loss:.4f}"
        )

        return avg_sup_loss, avg_unsup_loss


    def _train_one_epoch_concept(
        self,
        epoch: int,
    ) -> Tuple[float, float]:
        'Train one concept-supervised contrastive epoch.'
        self.model.train()
        tr_sup, steps_sup = 0.0, 0

        select_ids = self.train_outputs.get("select_ids", [])
        if len(select_ids) == 0:
            self.logger.warning(f"[Epoch {epoch}] No samples selected for concept SupCon.")
            return 0.0, 0.0


        _, sup_loader = get_pseudo_dataloader(
            args=self.args,
            train_outputs=self.train_outputs,
            mode="pretrain",
        )

        for batch in tqdm(sup_loader, desc=f"Train Concept-SupCon (Ep={epoch})"):
            text = batch["text_feats"].to(self.device)
            video = batch["video_feats"].to(self.device)
            audio = batch["audio_feats"].to(self.device)
            labels = batch["label_ids"].to(self.device)

            views = self._embed_views(text, video, audio)

            loss = self.supcon(
                views,
                labels=labels,
                temperature=self.args.train_temperature_sup,
                device=self.device,
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            tr_sup += loss.item()
            steps_sup += 1

        avg_sup_loss = tr_sup / max(1, steps_sup)
        self.logger.info(
            f"[Epoch {epoch}] Concept SupCon Loss: {avg_sup_loss:.4f}"
        )


        return avg_sup_loss, 0.0


    def _refresh_seed_features_with_anchors(
        self,
        feats: np.ndarray,
        gidx: np.ndarray,
    ) -> np.ndarray:
        'Replace seed features with intent anchors when configured.'
        if not getattr(self, "use_intents_as_seeds", False):
            return feats

        if self.intent_anchors is None or not self.seed_gids_per_concept:
            return feats

        gid2pos = {int(g): i for i, g in enumerate(gidx)}

        num_applied = 0
        for cid, gid_list in self.seed_gids_per_concept.items():
            if not gid_list:
                continue

            gid = int(gid_list[0])

            if cid >= self.intent_anchors.shape[0]:
                continue
            if gid not in gid2pos:
                continue

            pos = gid2pos[gid]
            feats[pos] = self.intent_anchors[cid].astype(feats.dtype)
            num_applied += 1

        self.logger.info(
            f"[SeedRefresh] Overwrote {num_applied} seed features with intent anchors."
        )
        return feats


    def _train(self, args):
        'Run MCSP warm-up and concept-aware graph training.'
        t_end_to_end = time.perf_counter()
        self.model.to(self.device)


        t0 = time.perf_counter()
        pack = self._extract_feats("train")
        self._record_time(
            "train_initial_extract_features", time.perf_counter() - t0
        )
        feats, gidx, y_true = pack["feats"], pack["idx"], pack["y_true"]


        t0 = time.perf_counter()
        feats, gidx, y_true = self._pre_round_build_intent_anchors(feats, gidx, y_true)
        self._record_time("train_preround_call", time.perf_counter() - t0)

        if not self.seed_gids_per_concept:
            self.logger.warning(
                "[Train] seed_gids_per_concept is empty after pre-round; "
                "label propagation–based Stage 2 may not work as expected."
            )

        final_assign = None
        num_epochs = int(self.max_rounds)

        for epoch in trange(num_epochs, desc="Epoch (Graph Stage-2)"):
            epoch_t0 = time.perf_counter()
            self.logger.info(f"--- Start Epoch {epoch} (Graph Concept-Label Stage 2) ---")

            N = feats.shape[0]


            t0 = time.perf_counter()
            feats = self._refresh_seed_features_with_anchors(feats, gidx)
            t_refresh_seed = time.perf_counter() - t0
            N = feats.shape[0]


            t0 = time.perf_counter()
            neighbors, weights_geom = self._build_knn_graph(feats)
            t_knn = time.perf_counter() - t0


            t0 = time.perf_counter()
            P_concept = self._compute_concept_distribution_from_anchors(feats)
            t_pconcept = time.perf_counter() - t0


            t0 = time.perf_counter()
            weights = self._modulate_edge_weights_with_concepts(
                neighbors, weights_geom, P_concept
            )
            t_modulate = time.perf_counter() - t0


            t0 = time.perf_counter()
            Y, seed_mask = self._build_seed_label_matrix_and_mask(N, gidx)
            t_seedmat = time.perf_counter() - t0


            t0 = time.perf_counter()
            Q = self._label_propagation(neighbors, weights, Y, seed_mask)
            t_random_walk = time.perf_counter() - t0


            t0 = time.perf_counter()
            concept_ids = Q.argmax(axis=1).astype(np.int64)
            conf = Q.max(axis=1).astype(np.float32)

            conf_mean = float(conf.mean())
            self.logger.info(
                f"[GraphLP][Epoch {epoch}] "
                f"mean_conf={conf_mean:.4f}, "
                f"min_conf={float(conf.min()):.4f}, max_conf={float(conf.max()):.4f}"
            )


            top_ratio = float(self.lp_conf_top_ratio)
            top_ratio = float(max(0.0, min(1.0, top_ratio)))

            if top_ratio <= 0.0:
                k_sel = N
                order = np.arange(N)
            else:
                k_sel = max(1, int(N * top_ratio))
                order = np.argsort(-conf)

            select_positions = order[:k_sel]
            select_gids = gidx[select_positions]

            self.logger.info(
                f"[Epoch {epoch}] Select by top-ratio: ratio={top_ratio:.3f}, k_sel={k_sel}"
            )


            self.train_outputs["label_ids"] = concept_ids
            self.train_outputs["select_ids"] = [int(g) for g in select_gids.tolist()]

            self.logger.info(
                f"[Epoch {epoch}] Concept-SupCon training set: "
                f"total N={N}, selected={len(select_gids)}, "
                f"conf_ratio={top_ratio:.3f}"
            )


            self.hq_ids = set(int(g) for g in select_gids.tolist())
            self.uq_ids = set(int(g) for g in gidx.tolist()) - self.hq_ids
            t_select_and_bookkeep = time.perf_counter() - t0


            t0 = time.perf_counter()
            sup_loss, _ = self._train_one_epoch_concept(epoch)
            t_train = time.perf_counter() - t0


            t0 = time.perf_counter()
            pack = self._extract_feats("train")
            feats, gidx, y_true = pack["feats"], pack["idx"], pack["y_true"]
            t_extract = time.perf_counter() - t0


            t0 = time.perf_counter()
            km_eval = KMeans(
                n_clusters=self.num_labels,
                init="k-means++",
                n_init=10,
                random_state=self.args.seed,
            ).fit(feats)
            assign_eval = km_eval.labels_
            final_assign = assign_eval
            t_kmeans = time.perf_counter() - t0

            metrics = clustering_score(y_true, assign_eval)
            acc = metrics.get("ACC", metrics.get("Accuracy", ""))
            nmi = metrics.get("NMI", "")
            ari = metrics.get("ARI", "")
            fmi = metrics.get("FMI", metrics.get("Fowlkes_Mallows", ""))

            self.logger.info(
                f"[Epoch {epoch}] Train clustering eval: "
                f"ACC={acc}, NMI={nmi}, ARI={ari}, FMI={fmi}"
            )

            epoch_total = time.perf_counter() - epoch_t0
            t_graph_rw_total = t_knn + t_pconcept + t_modulate + t_seedmat + t_random_walk


            self.logger.info(
                f"[Time][Epoch {epoch}] total={epoch_total:.3f}s | "
                f"graph+rw={t_graph_rw_total:.3f}s | random_walk={t_random_walk:.3f}s | "
                f"refresh_seed={t_refresh_seed:.3f}s | knn={t_knn:.3f}s | "
                f"pconcept={t_pconcept:.3f}s | modulate={t_modulate:.3f}s | "
                f"seedmat={t_seedmat:.3f}s | "
                f"select={t_select_and_bookkeep:.3f}s | train={t_train:.3f}s | "
                f"extract={t_extract:.3f}s | kmeans={t_kmeans:.3f}s"
            )
            self._record_time("epoch_total", epoch_total, epoch=epoch)
            self._record_time(
                "epoch_graph_rw_total", t_graph_rw_total, epoch=epoch
            )
            self._record_time("epoch_random_walk", t_random_walk, epoch=epoch)
            self._record_time("epoch_refresh_seed", t_refresh_seed, epoch=epoch)
            self._record_time("epoch_knn", t_knn, epoch=epoch)
            self._record_time("epoch_pconcept", t_pconcept, epoch=epoch)
            self._record_time("epoch_modulate", t_modulate, epoch=epoch)
            self._record_time("epoch_seedmat", t_seedmat, epoch=epoch)
            self._record_time(
                "epoch_select_bookkeep", t_select_and_bookkeep, epoch=epoch
            )
            self._record_time("epoch_concept_train", t_train, epoch=epoch)
            self._record_time("epoch_extract_features", t_extract, epoch=epoch)
            self._record_time("epoch_kmeans_eval", t_kmeans, epoch=epoch)


            with open(self.metrics_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                hq_size = len(select_gids)
                uq_size = N - hq_size
                writer.writerow(
                    [
                        epoch,
                        1.0,
                        hq_size,
                        uq_size,
                        sup_loss,
                        0.0,
                        acc,
                        nmi,
                        ari,
                        fmi,

                        epoch_total,
                        t_graph_rw_total,
                        t_random_walk,
                        t_refresh_seed,
                        t_knn,
                        t_pconcept,
                        t_modulate,
                        t_seedmat,
                        t_select_and_bookkeep,
                        t_train,
                        t_extract,
                        t_kmeans,
                    ]
                )


        if getattr(self.args, "save_model", False):
            t0 = time.perf_counter()
            save_model(self.model, self.args.model_output_path)
            self._record_time("save_model", time.perf_counter() - t0)

        if final_assign is None:
            final_assign = np.zeros_like(gidx, dtype=np.int64)

        t0 = time.perf_counter()
        self._export_final_hq(feats, gidx, final_assign)
        self._record_time("export_final_hq", time.perf_counter() - t0)
        self._record_time(
            "train_end_to_end_total", time.perf_counter() - t_end_to_end
        )
        self._write_timing_summary()


    def _export_final_hq(self, feats, idx, assign):
        'Export final cluster assignments and selected samples.'
        out_path = os.path.join(self.args.model_output_path, "final_hq_icr_id.csv")
        gid2pos = {int(gid): i for i, gid in enumerate(idx)}

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["gid", "cluster_id", "is_hq"])
            for gid in idx:
                gid = int(gid)
                pos = gid2pos[gid]
                cid = assign[pos]
                is_hq = 1 if gid in self.hq_ids else 0
                writer.writerow([gid, cid, is_hq])
        self.logger.info(f"Exported final HQ info to {out_path}")


    def _test(self, args):
        'Cluster test features and report evaluation results.'
        self.model.to(self.device)
        pack = self._extract_feats("test")
        feats, y_true = pack["feats"], pack["y_true"]

        km = KMeans(
            n_clusters=self.num_labels,
            init="k-means++",
            random_state=args.seed,
        ).fit(feats)
        y_pred = km.labels_

        artifacts_dir = os.path.join(args.pred_output_path, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        np.save(os.path.join(artifacts_dir, "y_pred.npy"), y_pred)
        import pickle
        feature_path = os.path.join(
            artifacts_dir, f"{args.dataset}_{args.method}_features.pkl"
        )
        with open(feature_path, "wb") as f:
            item={}
            item['feat']=feats
            item['y_pred']=y_pred
            pickle.dump(item,f)

        results = clustering_score(y_true, y_pred)
        self.logger.info("***** Test results *****")
        for k in sorted(results.keys()):
            self.logger.info("  %s = %s", k, str(results[k]))

        pred_labels_aligned, mapping = hungray_aligment(y_true, y_pred)
        matchings = dict(pred_labels_aligned)
        aligned_preds = np.array([matchings[p] for p in y_pred])
        print("++++++++++",y_pred)
        pred_labels_aligned=dict(pred_labels_aligned)
        print("=============",y_true)
        print("=============",pred_labels_aligned)
        print("=========",args.label_map)
        self.map_id2label = {v: k for k, v in args.label_map.items()}
        visualize_epoch_clustering_withid(args,
            feats,
            y_pred,
            y_pred,
            epoch=None,
            output_dir=os.path.join(artifacts_dir, "plots"),
            metrics=None,
            id2label=self.map_id2label
        )

        return results


    def _load_pretrained_backbone(self, pretrained_model):
        'Load pretrained backbone weights without task heads.'
        pretrained_dict = pretrained_model.state_dict()
        drop_keys = {
            "method_model.mlp_head_train.2.weight",
            "method_model.mlp_head_train.2.bias",
            "method_model.classifier.weight",
            "method_model.classifier.bias",
        }
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k not in drop_keys
        }
        print("sbsbsbbbbbbbbbbbbbbbbbbbbbbb")
        self.model.load_state_dict(pretrained_dict, strict=False)
