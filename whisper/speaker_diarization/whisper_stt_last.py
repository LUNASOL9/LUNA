"""Whisper + pyannote 3.1: 단어별 배정, 검토 기록, Excel용 CSV.
기존 venv에서 실행: python whisper_stt_improved.py
"""

import csv
import gc
import json
import math
import os
from pathlib import Path

AUDIO = Path(r"C:\LUNA\whisper\audio\Forward_Base_Taking_a_Day_Off.mp3")
FFMPEG_BIN = Path(r"C:\LUNA\ffmpeg-2026-10-15-git-afead44165-full_build\bin")
NUM_SPEAKERS = 4  # 녹음 전체 기준. 모르면 None
LANGUAGE = "korean"  # 자동 감지는 None
MODEL_ID = "openai/whisper-large-v3-turbo"


def save_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def valid_time(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def assign_words(chunks, segments):
    """STT 항목을 모두 보존하고 겹치는 화자가 없으면 UNKNOWN으로 남긴다."""
    assigned = []
    for chunk in chunks:
        start, end = chunk.get("timestamp") or (None, None)
        row = dict(
            start=start,
            end=end,
            text=chunk.get("text", ""),
            speaker_id="UNKNOWN",
            segment_index="",
            status="invalid_timestamp",
            overlap_ratio="",
            candidate_speakers="",
        )
        if valid_time(start) and valid_time(end) and end > start:
            hits = []
            for i, seg in enumerate(segments):
                overlap = max(0.0, min(end, seg["end"]) - max(start, seg["start"]))
                if overlap > 0:
                    hits.append((overlap, i, seg["speaker_id"]))
            row["status"] = "no_overlap"
            if hits:
                overlap, i, speaker = max(hits, key=lambda hit: hit[0])
                speakers = sorted({hit[2] for hit in hits})
                ratio = overlap / (end - start)
                row.update(
                    speaker_id=speaker,
                    segment_index=i,
                    status="multiple_speakers"
                    if len(speakers) > 1
                    else ("partial_overlap" if ratio < 0.5 else "matched"),
                    overlap_ratio=round(ratio, 4),
                    candidate_speakers=";".join(speakers),
                )
        assigned.append(row)
    return assigned


def build_final(segments, words):
    # 원래 발화 구간과 검토용 빈칸을 유지한다.
    rows = [
        dict(
            start=s["start"],
            end=s["end"],
            speaker_id=s["speaker_id"],
            duration=round(s["end"] - s["start"], 3),
            text="",
            status="no_stt_assigned",
            review_word_count=0,
        )
        for s in segments
    ]
    for word in words:
        i = word["segment_index"]
        if i == "":
            start, end = word["start"], word["end"]
            rows.append(
                dict(
                    start=start,
                    end=end,
                    speaker_id="UNKNOWN",
                    duration=round(end - start, 3)
                    if valid_time(start) and valid_time(end) and end >= start
                    else "",
                    text=word["text"],
                    status=word["status"],
                    review_word_count=1,
                )
            )
        else:
            rows[i]["text"] += word["text"]  # 원래 단어 사이 공백 유지
            rows[i]["status"] = "assigned"
            if word["status"] != "matched":
                rows[i]["review_word_count"] += 1
    for row in rows:
        row["text"] = row["text"].strip()
    return sorted(
        rows, key=lambda row: row["start"] if valid_time(row["start"]) else float("inf")
    )


def main():
    # 모델 라이브러리는 실행할 때 불러온다
    import torch
    import torchaudio
    from dotenv import load_dotenv
    from pyannote.audio import Pipeline
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from torch.torch_version import TorchVersion
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    load_dotenv()
    token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(".env에 HUGGING_FACE_TOKEN을 설정해 주세요.")
    if not AUDIO.is_file():
        raise FileNotFoundError(AUDIO)
    if FFMPEG_BIN.is_dir():
        os.environ["PATH"] += os.pathsep + str(FFMPEG_BIN)
    # 매번 별도 폴더에 저장하여 이전 결과를 보존
    from datetime import datetime

    out = AUDIO.parent / (
        AUDIO.stem + "_result_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    out.mkdir()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print("사용 장치:", device, flush=True)
    print("[1/4] 오디오 읽기", flush=True)
    waveform, sr = torchaudio.load(str(AUDIO))
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    sr = 16000
    if waveform.numel() == 0:
        raise ValueError("오디오가 비어 있습니다.")

    print("[2/4] Whisper 단어별 STT", flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )
    generation = {"task": "transcribe"}
    if LANGUAGE:
        generation["language"] = LANGUAGE
    result = asr(
        {"raw": waveform[0].numpy(), "sampling_rate": sr},
        return_timestamps="word",
        generate_kwargs=generation,
    )
    (out / "stt_raw.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    chunks = result.get("chunks", [])
    if not chunks and result.get("text"):
        # 시간 정보가 없어도 인식 문장은 보존
        chunks = [{"timestamp": (None, None), "text": result["text"]}]
    del asr, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[3/4] 화자 분리", flush=True)
    # 기존 환경의 로딩 방식을 유지 (pyannote.audio 3.x 대상)
    with torch.serialization.safe_globals(
        [TorchVersion, Specifications, Problem, Resolution]
    ):
        diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
    if diarizer is None:
        raise RuntimeError("pyannote 모델 접근 권한과 토큰을 확인해 주세요.")
    diarizer.to(torch.device(device))
    kwargs = {} if NUM_SPEAKERS is None else {"num_speakers": NUM_SPEAKERS}
    annotation = diarizer(
        {"waveform": waveform, "sample_rate": sr, "uri": AUDIO.stem}, **kwargs
    )
    with (out / "speakers.rttm").open("w", encoding="utf-8") as f:
        annotation.write_rttm(f)
    segments = sorted(
        [
            dict(start=float(turn.start), end=float(turn.end), speaker_id=str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ],
        key=lambda s: (s["start"], s["end"], s["speaker_id"]),
    )
    save_csv(out / "speaker_segments.csv", segments, ["start", "end", "speaker_id"])
    print("[4/4] 단어별 화자 배정 및 저장", flush=True)
    words = assign_words(chunks, segments)
    columns = [
        "start",
        "end",
        "speaker_id",
        "text",
        "segment_index",
        "status",
        "overlap_ratio",
        "candidate_speakers",
    ]
    save_csv(out / "words.csv", words, columns)
    review = [word for word in words if word["status"] != "matched"]
    save_csv(out / "review.csv", review, columns)
    final = build_final(segments, words)
    save_csv(
        out / (AUDIO.stem + "_by_Whisper_stt_Final.csv"),
        final,
        [
            "start",
            "end",
            "speaker_id",
            "duration",
            "text",
            "status",
            "review_word_count",
        ],
    )
    print(
        f"완료: {out}\nSTT 항목 {len(words)}개 / 검토 필요 {len(review)}개", flush=True
    )
    print("UNKNOWN은 배정 불가, no_stt_assigned는 붙일 STT가 없는 화자 구간입니다.")


if __name__ == "__main__":
    main()
