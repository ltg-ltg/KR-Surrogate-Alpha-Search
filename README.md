# 롱온리·고비용 시장에서 대리 모델 기반 알파 탐색 — 한국 주식시장

독립 연구. 선행 연구 Study 1(레포 `Surrogate-Alpha-Search`, 태그 `v1.0-results`)은 동기와 가설의 출처로만 인용한다.

## 상태: 완료 (2026-11-21)

- 사전 등록 `kr-prereg` (2026-11-20) → 단계 B 실행 → OOS 1회 (2026-11-20, 잠금). 단계 C는 생략했다
- **주 결론: 가설 기각.** R1·R2는 사전 등록한 방향의 반대로 실패했다 (원시 타깃 모델이 무작위보다 고유 통과를 더 많이 찾았고, 잔차 타깃은 원시보다 적게 찾았다). R3은 비열등이었다
- 결과: [docs/stageB_results.md](docs/stageB_results.md) (IS·Test 상세 [docs/stageB_is_test.md](docs/stageB_is_test.md)), 단계 A: [docs/stageA_report.md](docs/stageA_report.md)
- 사전 등록: [docs/preregistration.md](docs/preregistration.md)

## 문서

- 설계서: [docs/korea_study_design.md](docs/korea_study_design.md) (판정 규칙은 이 문서가 우선)
- 진행 체크리스트: [docs/korea_study_checklist.md](docs/korea_study_checklist.md)
- 선행 연구 비교: [docs/선행 연구 비교 정리 — 한국 롱온리 대리 모델 알파 탐색.md](<docs/선행 연구 비교 정리 — 한국 롱온리 대리 모델 알파 탐색.md>)

## 구간 (설계서 6절)

| 구간 | 기간 | 역할 |
|---|---|---|
| 버퍼 | 2010 | lookback 전용 |
| IS | 2011~2017 | 학습 타깃, 통과 기준 ①③④, 리밸런싱 주기 결정 |
| Test | 2018~2019 | 통과 기준 ② |
| OOS | 2020~2023 | 단계 B에서 처음 연다 |
| 예비 | 2024~ | **쓰지 않는다.** 데이터 폴더에 두지 않고 수집 코드도 받지 않는다 |

## 데이터

- 출처: **KRX 통계정보** (KRX Open API). 유가증권·코스닥 일별매매정보, 유가증권·코스닥 종목기본정보
- KRX 약관상 제3자 제공이 금지돼 원시·가공 시세는 이 레포에 없다. 수집·가공 코드와 결과(지표, 그림)만 둔다
- 데이터 위치: 레포 밖 `C:\kr_surrogate_alpha_search_data` (환경 변수 `KRSAS_DATA_ROOT`로 변경)
  - `raw/krx/{service}/{YYYY}/{YYYYMMDD}.json.gz`: API 응답 `OutBlock_1` 그대로
  - `raw/krx/_manifest.jsonl`: 파일별 sha256, 크기, 수집일, 엔드포인트, 수집 경로
  - `raw/krx/_calendar.json`: 평일별 거래일 여부

### 재현

1. KRX Open API 키를 발급받고 위 4개 서비스를 이용 신청한다
2. `.env.example`을 `.env`로 복사해 `KRX_OPEN_API`를 채운다
3. `python -m krsas.ingest.collect` (하루 한도 안에서 나눠 받는다. 약 3,450거래일 × 4개 서비스)
4. 공개된 원본 해시 목록과 대조한다 (해시 목록은 결과 공개 때 올린다)
5. 가공: `python -m krsas.ingest.clean` → `python -m krsas.ingest.adjust` (기본 공식) → `python -m krsas.ingest.dart`
   (OpenDART 키, crno 대응표) → `python -m krsas.ingest.rights --build-clean` → `python -m krsas.ingest.reset`
   (설계서 5.5절 재설정일 처리, 사건표 `config/reset_events.csv`) → `python -m krsas.validation` (단계 A 데이터 검증)

### 작성자 환경의 원본

작성자는 같은 키로 2026-11-16~19에 받아 둔 원본(kr_quant_machine `src/ingest/krx_collect.py`, 커밋 84c811d)을
`python -m krsas.ingest.import_local`로 옮겼다. 연구 기간 밖 파일은 옮기지 않았고, 파일 내용은 바꾸지 않았다.

## Study 1 코드와 수식 풀

- Study 1 레포는 `third_party/surrogate-alpha-search`에 git submodule로 두고 태그 `v1.0-results`(d2338b4)에 고정한다.
  받을 때: `git clone --recurse-submodules` 또는 `git submodule update --init`
- Study 1 코드가 최상위 패키지 `src`를 쓰므로 이 연구 패키지는 `krsas`다. `krsas.study1.ensure()`가 submodule을
  경로에 넣고 고정 커밋인지 확인한다. 문법·정규형·연산자는 Study 1 것을 그대로 쓰고, 백테스트 엔진은 롱온리로 새로 짠다
- 수식 풀 `pool_v1`(4만 개)은 Study 1 레포에 커밋돼 있지 않아 데이터 폴더 `study1/pool_v1.parquet`에 두고
  `krsas.pool`의 파일 해시·내용 해시로 고정한다. Study 1 v1.0-results 코드로 다시 생성한 풀과의 대조는 `python -m krsas.pool --compare <파일>`
- 설치: `python -m venv .venv` → `.venv/Scripts/python -m pip install -r requirements.txt` (Study 1 의존성 포함)

## 규칙

- 사전 등록(`kr-prereg`) 뒤의 변경은 모두 "탐색"으로 표시한다
- OOS 결과 파일은 `krsas.oos`만 읽는다 (`tests/test_oos_lock.py`). OOS는 이미 한 번 실행해 잠겼다
- 연구 기간 밖 날짜는 `krsas.paths.check_date`가 막는다
