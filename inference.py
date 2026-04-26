# inference.py
# Core inference engine — loaded once at startup, reused per request

import os
import cv2
import pickle
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


# ── Constants ────────────────────────────────────────────────────────
CLASSES = ["Aggressive_Hyperactive", "Pacing_Anxious", "Resting_Lethargic"]

FLUTTER_LABEL = {
    "Aggressive_Hyperactive": "Aggressive / Hyperactive",
    "Pacing_Anxious":         "Anxiety",
    "Resting_Lethargic":      "Resting / Lethargic",
}

IMAGE_SIZE  = 224
EXTRACT_FPS = 3.0
W_XGB       = 0.55
W_CNN       = 0.45

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

PREPROCESS = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


# ── Keyword → class mapping (text-only path) ─────────────────────────
TEXT_KEYWORDS = {
    "Aggressive_Hyperactive": [
        "aggressive", "aggression", "biting", "bite", "growling", "growl",
        "lunging", "lunge", "barking", "bark", "snapping", "snap",
        "hyperactive", "hyper", "zoomies", "attacking", "attack",
        "snarling", "snarl", "charging", "wild", "frantic", "manic",
        "destructive", "destroy", "chewing everything",
    ],
    "Pacing_Anxious": [
        "pacing", "pace", "anxious", "anxiety", "restless", "circling",
        "circle", "whining", "whine", "trembling", "tremble", "shaking",
        "shake", "nervous", "stress", "stressed", "separation",
        "hiding", "hide", "clingy", "cling", "excessive barking",
        "panting", "pant", "fearful", "fear", "scared", "scared",
        "cowering", "cower", "yowling", "yowl", "won't stop moving",
        "keeps walking", "back and forth",
    ],
    "Resting_Lethargic": [
        "lethargic", "lethargy", "tired", "sleepy", "sleeping too much",
        "not moving", "won't move", "inactive", "lazy", "depressed",
        "depression", "sad", "not eating", "loss of appetite",
        "withdrawn", "unresponsive", "low energy", "barely moving",
        "just lying", "not playing", "no interest", "dull",
    ],
}


def classify_text(text: str) -> tuple[str, float]:
    """
    Keyword-based classification for text-only input.
    Returns (class_name, confidence_score).
    """
    text_lower = text.lower()
    scores = {cls: 0 for cls in CLASSES}

    for cls, keywords in TEXT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cls] += 1

    total = sum(scores.values())
    if total == 0:
        # No keywords matched — return unknown
        return None, 0.0

    best_cls = max(scores, key=scores.get)
    confidence = scores[best_cls] / total

    # Normalize to a reasonable confidence range (0.55 – 0.85)
    confidence = 0.55 + (confidence * 0.30)
    return best_cls, round(min(confidence, 0.85), 4)


# ── Suggestions lookup ───────────────────────────────────────────────
SUGGESTIONS = {
    "Aggressive_Hyperactive": {
        "low": [
            "Mild excitement or play aggression detected.",
            "Redirect energy immediately with an appropriate toy.",
            "Avoid rough play that increases arousal levels.",
            "End interactions calmly if teeth come out.",
        ],
        "medium": [
            "Moderate aggression or hyperactivity detected.",
            "Identify and remove the trigger from the environment.",
            "Never punish aggressive behaviour — it escalates the response.",
            "A 20–30 min structured exercise session before alone time helps.",
            "Schedule a vet check to rule out pain-induced aggression.",
        ],
        "high": [
            "High aggression detected — prioritise safety immediately.",
            "Separate your pet from the trigger environment safely.",
            "Do not approach an aggressive pet without precautions.",
            "Book an urgent veterinary appointment — pain is a common hidden cause.",
            "A certified animal behaviourist consultation is strongly recommended.",
        ],
    },
    "Pacing_Anxious": {
        "low": [
            "Mild restlessness or anxiety detected.",
            "Check for environmental stressors such as loud sounds or changes at home.",
            "Provide a quiet safe space with familiar bedding or a worn garment.",
            "Maintain a consistent daily feeding and walk schedule.",
        ],
        "medium": [
            "Moderate anxiety and pacing behaviour detected.",
            "Establish a predictable daily routine — it significantly reduces anxiety.",
            "Try a calming diffuser: Adaptil for dogs or Feliway for cats.",
            "Practice short departure exercises to build confidence gradually.",
            "Consult your vet if pacing persists beyond 3 consecutive days.",
        ],
        "high": [
            "Significant anxiety detected — your pet needs intervention now.",
            "Move your pet to the calmest, most familiar room immediately.",
            "Contact a vet or certified animal behaviourist within 24–48 hours.",
            "Anti-anxiety support medication may be appropriate — discuss with your vet.",
            "Never restrain an anxious pet — it significantly worsens the condition.",
        ],
    },
    "Resting_Lethargic": {
        "low": [
            "Your pet appears calm and resting normally.",
            "Ensure fresh water is always accessible.",
            "No immediate action needed — this is healthy behaviour.",
            "Continue regular exercise and enrichment sessions.",
        ],
        "medium": [
            "Your pet is showing mild lethargy signs.",
            "Monitor eating and drinking patterns closely over the next 24 hours.",
            "A short gentle 10-minute play or walk session can improve mood.",
            "Ensure the rest area is comfortable and at an appropriate temperature.",
        ],
        "high": [
            "Significant lethargy detected — veterinary attention recommended.",
            "Check for vomiting, diarrhoea, or laboured breathing.",
            "Reduced appetite combined with lethargy is a red flag — see a vet soon.",
            "Keep your pet warm, calm, and hydrated until seen by a professional.",
            "Do not delay — lethargy can be an early sign of serious illness.",
        ],
    },
}

# ── Breed-specific context notes ─────────────────────────────────────
BREED_CONTEXT = {
    ("Resting_Lethargic",      "ragdoll"):           "Ragdolls naturally go limp — extended lethargy still warrants monitoring.",
    ("Resting_Lethargic",      "persian"):            "Persians are calm by nature — only concerning if combined with appetite loss.",
    ("Resting_Lethargic",      "british_shorthair"):  "British Shorthairs are naturally calm — watch for appetite changes.",
    ("Pacing_Anxious",         "siamese"):            "Siamese are naturally vocal and active — cross-check with other anxiety signs.",
    ("Pacing_Anxious",         "german_shepherd"):    "GSDs bond intensely — pacing alone is a common separation anxiety sign.",
    ("Aggressive_Hyperactive", "bully_kutta"):        "Bully Kuttas have strong guarding instincts — identify the trigger first.",
    ("Aggressive_Hyperactive", "bengal"):             "Bengal zoomies are normal high-energy bursts — check duration and context.",
    ("Pacing_Anxious",         "golden_retriever"):   "Golden Retrievers are social — pacing often signals loneliness or boredom.",
    ("Aggressive_Hyperactive", "rottweiler"):         "Rottweilers are protective — distinguish guarding behaviour from aggression.",
    ("Resting_Lethargic",      "labrador"):           "Labradors are naturally energetic — unusual lethargy warrants a vet check.",
}


class PetBehaviorEngine:
    """
    Two-stream inference engine.
    Stream 1 : Optical flow features  →  XGBoost
    Stream 2 : RGB keyframes           →  TorchScript CNN
    Ensemble  : Weighted average of both probability vectors
    Text path : Keyword scoring (no model needed)
    """

    def __init__(
        self,
        xgb_path:    str,
        scaler_path: str,
        cnn_path:    str,
    ):
        # Load XGBoost
        with open(xgb_path, "rb") as f:
            self.xgb = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load TorchScript CNN
        self.device = torch.device("cpu")   # Railway free tier — CPU only
        self.cnn    = torch.jit.load(cnn_path, map_location=self.device)
        self.cnn.eval()

        print("PetBehaviorEngine loaded successfully.")
        print(f"  XGBoost  : {xgb_path}")
        print(f"  CNN      : {cnn_path}")
        print(f"  Device   : {self.device}")

    # ── Optical flow features ─────────────────────────────────────────
    def _flow_features(self, video_path: str) -> np.ndarray:
        cap      = cv2.VideoCapture(video_path)
        vid_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(vid_fps / EXTRACT_FPS))

        prev_gray = None
        frame_idx = 0
        magnitudes, mags_all, angs_all, xs, ys = [], [], [], [], []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                gray = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (112, 112)
                )
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    mag, ang = cv2.cartToPolar(
                        flow[..., 0], flow[..., 1], angleInDegrees=True
                    )
                    magnitudes.append(float(mag.mean()))
                    mags_all.extend(mag.flatten().tolist())
                    angs_all.extend(ang.flatten().tolist())
                    xs.append(float(flow[..., 0].mean()))
                    ys.append(float(flow[..., 1].mean()))
                prev_gray = gray
            frame_idx += 1
        cap.release()

        if len(magnitudes) < 2:
            return np.zeros(28, dtype=np.float32)

        mags = np.array(magnitudes,  dtype=np.float32)
        am   = np.array(mags_all,    dtype=np.float32)
        aa   = np.array(angs_all,    dtype=np.float32)
        xsa  = np.array(xs,          dtype=np.float32)
        ysa  = np.array(ys,          dtype=np.float32)

        dh, _ = np.histogram(aa, bins=8, range=(0, 360))
        dhr   = dh.astype(np.float32)
        dhn   = dhr / (dhr.sum() + 1e-8)
        p     = dhn + 1e-8

        return np.array([
            mags.mean(), mags.std(), mags.max(), np.percentile(mags, 95),
            *dhr, *dhn,
            float(-np.sum(p * np.log2(p))),
            float(mags.var()),
            float(am.max()),
            float((am < 1.0).mean()),
            float((am > 5.0).mean()),
            float(-np.sum(dhn * np.log2(dhn + 1e-8))),
            float(xsa.mean()),
            float(ysa.mean()),
        ], dtype=np.float32)

    # ── CNN keyframe inference ────────────────────────────────────────
    def _cnn_probs(self, video_path: str) -> np.ndarray:
        cap     = cv2.VideoCapture(video_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_f - 1, min(8, total_f), dtype=int)
        tensors = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                tensors.append(PREPROCESS(pil))
        cap.release()

        if not tensors:
            return np.ones(len(CLASSES), dtype=np.float32) / len(CLASSES)

        batch  = torch.stack(tensors)        # (N, 3, H, W)
        with torch.no_grad():
            logits = self.cnn(batch)         # (N, 3)
        probs  = torch.softmax(logits, dim=1).numpy()
        return probs.mean(axis=0)            # average over keyframes

    # ── Main predict ──────────────────────────────────────────────────
    def predict(
        self,
        text:       str  = "",
        video_path: str  = None,
        breed:      str  = "unknown",
        animal:     str  = "unknown",
    ) -> dict:
        """
        Unified prediction supporting three input modes:
          1. Text only         →  keyword classification
          2. Video only        →  XGBoost + CNN ensemble
          3. Text + Video      →  ensemble + text context boost

        Returns the JSON dict your Flutter app parses.
        """
        has_text  = bool(text and text.strip())
        has_video = bool(video_path and os.path.exists(video_path))
        input_mode = []

        # ── Mode 1: Video path ────────────────────────────────────────
        if has_video:
            input_mode.append("video")

            # Stream 1 — optical flow → XGBoost
            feat      = self._flow_features(video_path).reshape(1, -1)
            feat_sc   = self.scaler.transform(feat)
            xgb_probs = self.xgb.predict_proba(feat_sc)[0]

            # Stream 2 — CNN
            cnn_probs = self._cnn_probs(video_path)

            # Weighted ensemble
            final_probs = W_XGB * xgb_probs + W_CNN * cnn_probs

            # If text also provided, use it to boost matching class
            if has_text:
                input_mode.append("text")
                text_cls, text_conf = classify_text(text)
                if text_cls:
                    boost = np.zeros(len(CLASSES), dtype=np.float32)
                    boost[CLASSES.index(text_cls)] = 0.15 * text_conf
                    final_probs = final_probs + boost
                    # Re-normalise
                    final_probs = final_probs / final_probs.sum()

        # ── Mode 2: Text only ─────────────────────────────────────────
        elif has_text:
            input_mode.append("text")
            text_cls, text_conf = classify_text(text)

            if text_cls:
                # Build a soft probability vector from keyword confidence
                final_probs = np.full(len(CLASSES), 0.10, dtype=np.float32)
                final_probs[CLASSES.index(text_cls)] = text_conf
                final_probs = final_probs / final_probs.sum()
            else:
                # No keywords matched — return unknown response
                return self._unknown_response(text)

        else:
            return self._unknown_response("")

        # ── Decode prediction ─────────────────────────────────────────
        pred_idx   = int(np.argmax(final_probs))
        confidence = float(final_probs[pred_idx])
        cls_name   = CLASSES[pred_idx]

        # Severity bands
        if confidence >= 0.80:
            severity = "high"
        elif confidence >= 0.55:
            severity = "medium"
        else:
            severity = "low"

        # Breed context
        breed_lower  = breed.lower().replace(" ", "_")
        breed_note   = BREED_CONTEXT.get((cls_name, breed_lower), "")
        suggestions  = list(SUGGESTIONS[cls_name][severity])
        if breed_note:
            suggestions = [f"Breed note: {breed_note}"] + suggestions

        # Early warning
        early_warning = (
            cls_name in {"Aggressive_Hyperactive", "Pacing_Anxious"}
            and severity in {"medium", "high"}
        )

        return {
            "success": True,
            "detected_behavior": FLUTTER_LABEL[cls_name],
            "behavior_key":      cls_name,
            "confidence":        round(confidence, 4),
            "confidence_percent": f"{round(confidence * 100)}%",
            "severity":          severity,
            "early_warning":     early_warning,
            "suggestions":       suggestions,
            "breed_context":     breed_note,
            "all_probabilities": {
                FLUTTER_LABEL[CLASSES[i]]: round(float(final_probs[i]), 4)
                for i in range(len(CLASSES))
            },
            "input_mode":    "+".join(input_mode),
            "animal":        animal,
            "breed":         breed,
            "model_version": "1.0.0",
        }

    def _unknown_response(self, text: str) -> dict:
        return {
            "success":           False,
            "detected_behavior": "Unable to determine",
            "behavior_key":      "unknown",
            "confidence":        0.0,
            "confidence_percent": "0%",
            "severity":          "low",
            "early_warning":     False,
            "suggestions": [
                "Please describe your pet's behaviour in more detail.",
                "Try uploading a short video clip for a more accurate diagnosis.",
                "Common signs to describe: pacing, hiding, aggression, lethargy.",
                "If your pet seems unwell, consult a vet as a first step.",
            ],
            "breed_context":     "",
            "all_probabilities": {
                "Aggressive / Hyperactive": 0.0,
                "Anxiety":                  0.0,
                "Resting / Lethargic":      0.0,
            },
            "input_mode":    "text" if text else "none",
            "model_version": "1.0.0",
        }
