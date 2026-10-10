# Data Chef AI — GPT-6 Luna Edition

**한 개의 완성형 API 기반 수업 프로젝트** · Python + Streamlit + OpenAI Responses API

가장 빠른 시작: [START_HERE_먼저읽기.md](START_HERE_먼저읽기.md)

## 무엇이 구현되어 있나요?

- 사용자 보유 재료(한글·대략적 수량)와 조리 시간·목표·제외 알레르겐 입력
- API 키가 설정되면 **`gpt-6-luna`가 1인분 새 레시피 4개 제안**
- AI 후보의 사용 식재료·수량·제외 제약 조건을 Python이 검증
- 검토된 12개 기본 레시피도 함께 100점 알고리즘으로 비교 추천
- 데이터 기반 영양소 계산, 부족 재료 표시, 추천 이유 및 조리 과정, 5축 레이더 차트
- 냉장고 사진에서 식재료 읽기 (결과를 직접 확인·수정해 레시피 탭으로 보낼 수 있음)
- 식품 포장지 사진에서 영양성분·원재료·알레르기 표시 추출
- 재료 궁합 예시 행렬, 가중치 민감도 실험, 자체 영양성분 CSV 사용
- 점수·입력·추천 레시피·AI 분석 정보 JSON 저장
- API 비밀키는 환경변수 / Streamlit Secrets / 로컬 입력에서 읽으며 소스에 저장하지 않음
- 데모용 접근 코드로 유료 API 사용량 보호 가능

## GPT와 Python의 역할 구분

| 단계 | 담당 | 검증 |
|---|---|---|
| 요리 이름·조리 순서·구성 제안 | GPT-6 Luna Responses API | 재료/수량/제약 조건 검사 |
| 알레르기·사용금지 재료 제한 | Python | 등록된 사전에 대한 조건 필터 |
| 영양량 계산 | Python | 100g 단위 × g/100 |
| 추천 점수 및 최종 순위 | Python | 100점 상대 점수 |
| 영양 차트 | Plotly | 비교를 위한 단순 목표 충족률 |
| 포장식품 라벨 읽기 | GPT-6 Luna Vision | 불확실성 기록, 원본 대조 권장 |
| 냉장고 사진에서 식재료 인식 | GPT-6 Luna Vision | 사진만으로 수량/신선도 단정 안 함 |

모든 GPT 호출은 Responses API(`client.responses.create`)를 사용하고, Structured Outputs의 엄격한 JSON Schema를 지정합니다. `store=False`로 설정했습니다. 모델 이름은 코드에 고정되어 있어 메뉴에서 다른 모델로 변경되지 않습니다.

## 프로젝트 디렉터리

```text
app.py                     Streamlit UI
chef_core.py               로컬 추천 엔진·수량 변환·영양 계산
ai_service.py              GPT-6 Luna API 및 응답 검증
requirements.txt           배포시 자동 의존성 설치
.streamlit/config.toml     사이트 테마
.streamlit/secrets.toml.example  API 키 예시(실제 키 없음)
data/demo_nutrition.csv    시연용 100g 단위 영양 데이터
  /recipes.json            검토용 레시피 12개
  /compatibility.csv       시연용 재료 궁합 행렬
tests/                     인터넷 없이 실행 가능한 자동 테스트
START_HERE_먼저읽기.md      실행 안내
REPORT_GUIDE.md            과제 제출 구상
run_windows.bat            로컬 Windows 실행
```

## 개발자 확인 명령

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

## 정확도 · 안전 · 한계

기본 포함 100g 영양 데이터는 제품 브랜드/가공상태/조리 손실까지 반영한 정밀 DB가 아닌 **프로젝트 시연용 근사치**입니다. 궁합 행렬도 맛/조리 적합성에 대한 가상 예시일 뿐 영양학적 상승작용을 주장하지 않습니다. 실생활에서 섭취 안전성은 보관 상태·알레르기·포장 표시를 우선 확인해야 합니다. 업로드한 공식 DB와 앱 기본 DB가 100g 기준이 다른 경우 직접 정제해야 합니다. 이 도구는 의료/식이 처방 시스템이 아닙니다.

레시피 생성은 모델 품질과 API 키 사용 권한에 영향을 받습니다. 잘못된 후보는 제외하거나 기본 레시피로 넘어가며 모델 제안의 모든 안전성을 기계적으로 보증할 수는 없습니다.

## 공식 기술 문서

- [GPT-6 Luna 모델 ID](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Images and Vision](https://developers.openai.com/api/docs/guides/images-vision)
- [Streamlit Cloud Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
