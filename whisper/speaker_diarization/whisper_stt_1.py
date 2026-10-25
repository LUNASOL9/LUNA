import os
import torch
import pandas as pd
import torchaudio

from transformers import AutoModelForSpeechSeq2Seq,  AutoProcessor, pipeline
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Specifications, Problem, Resolution
from torch.torch_version import TorchVersion
from dotenv import load_dotenv

load_dotenv()

os.environ["PATH"] += os.pathsep + r"C:\LUNA\ffmpeg-2026-10-15-git-afead44165-full_build\bin"

def whisper_stt(
  audio_file_path: str,
  output_file_path: str = "./output.csv"      
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
            torch_dtype=torch_dtype, 
                low_cpu_mem_usage=True, 
                    use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True, # 청크별로 타임스탬프를 반환
    chunk_length_s=10,      # 입력 오디오를 10초씩 나누기
    stride_length_s=2,      # 청크가 2초씩 겹치도록 나누기
    )
    
    result = pipe(audio_file_path)
    df =  whisper_to_dataframe(result, output_file_path)

    return result, df




# CSV 파일로 저장한 후 데이터프레임을 반환
def whisper_to_dataframe(result, output_file_path):
    start_end_text = []

    for chunk in result["chunks"]:
        start = chunk["timestamp"][0]
        end = chunk["timestamp"][1]
        text = chunk["text"].strip()
        start_end_text.append([start, end, text])
        df = pd.DataFrame(start_end_text, columns=["start", "end", "text"])
        df.to_csv(output_file_path, index=False, sep="|", encoding="utf-8-sig")

    return df




# 화자 분리
def speaker_diarization(
   audio_file_path: str,
   output_rttm_file_path: str,
   output_csv_file_path: str
   ) :
    

    allowed_classes = [TorchVersion, Specifications, Problem, Resolution]

    with torch.serialization.safe_globals(allowed_classes):
        pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HUGGING_FACE_TOKEN"),
    )

    # CUDA를 사용할 수 있다면 CUDA를 사용하도록 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print(f"사용 장치: {device}")
    
    # MP3を一度だけ読み込む
    waveform, sample_rate = torchaudio.load(audio_file_path)

    # モノラル・16kHzに統一
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, 16000
        )
        sample_rate = 16000

    diarization_pipeline = pipeline({
        "waveform": waveform,
        "sample_rate": sample_rate,
        "uri": audio_file_path,
    })
    
    # dump the diarization output to disk using RTTM format
    with open(output_rttm_file_path, "w", encoding='utf-8') as rttm:
        diarization_pipeline.write_rttm(rttm)

    # pandas dataframe으로 변환
    df_rttm = pd.read_csv(
        output_rttm_file_path,      # rttm 파일 경로
        sep=' ',        # 구분자는 띄어쓰기
        header=None,    # 헤더는 없음
        names=['type', 'file', 'chnl', 'start', 'duration', 'C1', 'C2', 'speaker_id', 'C3', 'C4'] 
    )
    
    df_rttm["end"] = df_rttm["start"] + df_rttm["duration"]

    # speaker_id를 기반으로 화자별로 구간을 나누기
    df_rttm["number"] = None
    df_rttm.at[0, "number"] = 0

    for i in range(1, len(df_rttm)):
        if df_rttm.at[i, "speaker_id"] != df_rttm.at[i-1, "speaker_id"]:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"] + 1
        else:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"]

    df_rttm_grouped = df_rttm.groupby("number").agg(
        start=pd.NamedAgg(column='start', aggfunc='min'),
        end=pd.NamedAgg(column='end', aggfunc='max'),
        speaker_id=pd.NamedAgg(column='speaker_id', aggfunc='first')
    )

    df_rttm_grouped["duration"] = df_rttm_grouped["end"] - df_rttm_grouped["start"]

    df_rttm_grouped.to_csv(
        output_csv_file_path,
        index=False,    # 인덱스는 저장하지 않음
        encoding='utf-8' 
    )
    return df_rttm_grouped




if __name__ == "__main__":
    audio_file_path =  r"C:\LUNA\whisper\audio\Forward_Base_Taking_a_Day_Off.mp3"                       # 원본 오디오 파일
    stt_output_file_path =  r"C:\LUNA\whisper\audio\Forward_Base_Taking_a_Day_Off_by_Whisper_stt.csv"	# STT 결과 파일
    rttm_file_path = r"C:\LUNA\whisper\audio\Forward_Base_Taking_a_Day_Off_by_Whisper_stt.rttm"		    # 화자 분리 원본 파일
    rttm_csv_file_path = r"C:\LUNA\whisper\audio\Forward_Base_Taking_a_Day_Off_by_Whisper_stt.rttm.csv"	# 화자 분리 CSV 파일
    
    df_rttm = speaker_diarization(
        audio_file_path,
        rttm_file_path,
        rttm_csv_file_path
    )
    print(df_rttm)