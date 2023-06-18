1. DEAM
DEAM dataset 및 전처리 중간결과물

* DEAM dataset download
https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music
 - audio: DEAM dataset에서 다운로드 받은 MP3 파일들
 - 15_CUT_MP3: 첫 15초를 자른 MP3 파일들
 - spleeter_WAV: 단일 MP3 파일에서 5개의 단일 악기 소리를 추출해 저장한 5개 WAV 파일들
 - WAV2MIDI: WAV에서 MIDI로 변환한 파일들
 - midi_files: 단일 악기 소리를 담은 MIDI 파일들을 multi-instrument로 병합한 MIDI 파일들
 - features: music_generator/create_dataset/run.py를 실행해서 생성된 json 및 csv 파일들
 - pt_files: Music Generator가 생성한 음악
 - arousal.csv, valence.csv: DEAM dataset에서 다운로드 받은 arousal.csv 및 valence.csv

2. XED
XED dataset
* XED dataset download
https://github.com/Helsinki-NLP/XED

3. IMSDB
IMSDb에서 추출한 dialogues 전처리 결과물
* IMSDb dataset link
https://imsdb.com/