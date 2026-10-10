"""Data Chef AI · one GitHub-ready Streamlit app · GPT-6 Luna only."""
from __future__ import annotations
import json
import os
import hashlib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_service import (MODEL, analyze_food_label, create_recipe_candidates,
                        recognize_fridge_photo, validate_ai_recipes)
from chef_core import (ALLERGENS, DEFAULT_WEIGHTS, DISPLAY_FIELDS, GOALS,
                       effective_weights, load_demo_nutrients, load_recipes,
                       normalize_official_csv, parse_inventory,
                       radar_target_ratio, rank_recipes)

st.set_page_config(page_title='Data Chef AI · Luna', page_icon='🥗', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family:'Nunito Sans',sans-serif;}
.block-container {max-width:1170px; padding-top:1.35rem;}
h1 {letter-spacing:-1.3px;} h2,h3 {letter-spacing:-.45px;}
[data-testid="stMetric"] {background:#eff5f1; border:1px solid #e0e9e2; padding:14px 17px; border-radius:14px;}
[data-testid="stSidebar"] {background:#eff5f1;}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] {border-radius:11px; font-weight:750;}
.hero {padding:22px 27px; border-radius:18px; background:linear-gradient(100deg,#e7f3e9,#f7f6e9);
        border:1px solid #d7e8dc; margin-bottom:22px;}
.eyebrow {color:#426f62; font-size:11px; font-weight:800; letter-spacing:2px;}
.hero-title {font-size:2.25rem;font-weight:800;color:#24493a;margin-top:5px;line-height:1.15;}
.hero-sub {color:#52685b;font-size:15px;margin-top:8px;}
.pill {display:inline-block;background:#dcecdf;color:#31624c;padding:5px 10px;border-radius:99px;font-size:12px;font-weight:700;}
.small-muted {color:#607a6b;font-size:12px;}
</style>''', unsafe_allow_html=True)


def get_secret(name: str, default='') -> str:
    env = os.getenv(name, '').strip()
    if env: return env
    try:
        return str(st.secrets.get(name,default)).strip()
    except Exception:
        return default


# Optional passcode protects paid model calls in a shared demo deployment.
access_code = get_secret('APP_ACCESS_CODE')
if access_code:
    st.title('🔐 Data Chef AI')
    st.info('이 배포판은 API 과금 보호를 위해 접근 코드가 설정되어 있어요.')
    entered = st.text_input('접근 코드',type='password')
    import hmac
    if not hmac.compare_digest(entered,access_code):
        st.stop()

with st.sidebar:
    st.markdown('## 🥗 Data Chef')
    st.caption('GPT-6 Luna · 데이터 기반 요리 연구실')
    st.divider()
    st.markdown('**AI 모델**')
    st.code(MODEL, language=None)
    saved_key=get_secret('OPENAI_API_KEY')
    local_key=st.text_input('API 키 (로컬 테스트용)',type='password',
                            help='배포할 때는 Streamlit Secrets에 넣으세요. 코드나 GitHub에 올리지 마세요.')
    api_key=local_key.strip() or saved_key
    if api_key:
        st.success('루나 API 사용 준비됨')
    else:
        st.warning('API 키 미설정 · 기본 추천 체험 가능')
        st.caption('키를 넣으면 루나가 새 레시피를 생성하고 사진도 분석합니다.')
    st.divider()
    st.markdown('**이 앱의 역할 분담**')
    st.caption('🧠 루나 : 레시피 창작 · 식재료 사진 인식 · 성분표 읽기')
    st.caption('🧮 Python : 영양 계산 · 제약 조건 검사 · 순위 결정')
    st.divider()
    st.caption('⚠️ 기본 영양 DB와 재료 궁합은 수업용 근사값입니다. 식품 안전·의료 판단이 아닙니다.')
    st.caption('📷 사진 분석 시 이미지가 OpenAI API로 전송됩니다.')

st.markdown('''<div class="hero">
<div class="eyebrow">SMART KITCHEN · FOOD DATA LAB</div>
<div class="hero-title">🥬 Data Chef AI</div>
<div class="hero-sub">냉장고 속 재료를 더 똑똑하게. 레시피 창작부터 영양 균형과 식품 성분표까지.</div>
<div style="margin-top:11px"><span class="pill">GPT-6 Luna</span> &nbsp; <span class="pill">데이터 기반 100점 추천</span> &nbsp; <span class="pill">영양 레이더</span></div>
</div>''', unsafe_allow_html=True)

tab_recipe,tab_photo,tab_label,tab_data=st.tabs(['🍳 레시피 추천','📸 냉장고 사진','🔎 성분표 분석','📊 데이터 / 알고리즘'])


def store_result(inventory, warnings, ranked, goal, weights, origin, validations):
    st.session_state['results'] = ranked
    st.session_state['inventory'] = inventory
    st.session_state['goal'] = goal
    st.session_state['weights'] = effective_weights(weights)
    st.session_state['warnings'] = warnings
    st.session_state['origin'] = origin
    st.session_state['validations'] = validations


def radar_chart(nutrition: dict, goal: str):
    ratios = radar_target_ratio(nutrition,goal)
    keys = list(ratios)
    theta = keys+[keys[0]]
    values = [ratios[k] for k in keys]+[ratios[keys[0]]]
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=[100]*len(theta),theta=theta, mode='lines',
                                  line=dict(color='#a4b4ab',dash='dot'),name='목표 100%'))
    fig.add_trace(go.Scatterpolar(r=values,theta=theta,fill='toself',
                                  name='레시피',line=dict(color='#238a71',width=3),
                                  fillcolor='rgba(35,138,113,.24)'))
    fig.update_layout(height=385,margin=dict(l=45,r=45,t=30,b=40),
                      legend=dict(orientation='h',y=-.18,x=.23),
                      paper_bgcolor='rgba(0,0,0,0)',
                      polar=dict(bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(range=[0,100],tickvals=[25,50,75,100],tickfont=dict(size=10))))
    return fig


with tab_recipe:
    st.markdown('### 01 · 내 냉장고 재료로 추천받기')
    st.caption('재료를 입력하면 루나가 새 레시피 후보를 만들고, Python이 실제 재고와 영양 데이터를 검증해 순위를 매깁니다.')
    if 'pending_ingredient_text' in st.session_state:
        st.session_state['ingredient_text']=st.session_state.pop('pending_ingredient_text')
    if 'ingredient_text' not in st.session_state:
        st.session_state['ingredient_text']='계란 2개, 두부 반 모, 양파 1개, 시금치 100g, 식용유 10g'
    with st.form('recipe_form',clear_on_submit=False):
        st.text_area('보유 식재료 및 수량',height=100,key='ingredient_text',
                     help='예: 계란 2개, 두부 150g, 김치 100g. g 단위 입력이 가장 정확해요.')
        c1,c2,c3=st.columns(3)
        with c1: goal=st.selectbox('오늘의 식사 목표',list(GOALS),index=1)
        with c2: max_minutes=st.slider('조리 가능한 시간',5,60,25,5)
        with c3: top_k=st.selectbox('추천 결과 개수',[3,5,8],index=0)
        c1,c2=st.columns(2)
        with c1: soon_text=st.text_input('먼저 써야 하는 재료 (선택)',placeholder='예: 시금치, 두부')
        with c2: bad_text=st.text_input('사용 금지 재료 (선택)',placeholder='예: 상한 우유, 계란')
        allergen_list=sorted(set().union(*ALLERGENS.values()))
        allergen_exclusions=st.multiselect('알레르기 / 회피 성분 제외',allergen_list,
                                          help='등록된 식재료 기준의 1차 필터입니다. 실제 식품 표시도 확인하세요.')
        with st.expander('⚖️ 점수 가중치 조정 (발표용 비교 실험)'):
            st.caption('합계는 자동으로 100점 기준으로 환산됩니다.')
            wc1,wc2=st.columns(2)
            weights={}
            for ix,(criterion,wt) in enumerate(DEFAULT_WEIGHTS.items()):
                with (wc1 if ix%2==0 else wc2):
                    weights[criterion]=st.slider(criterion,0,60,wt,5,key=f'weight_{ix}')
        do_ai=st.checkbox('루나가 새로운 레시피 후보 4개를 직접 만들기 (API 호출)',value=bool(api_key),
                          disabled=not bool(api_key),
                          help='선택하면 추천 버튼을 누를 때 1회 유료 API 호출합니다.')
        pressed=st.form_submit_button('✨ 맞춤 레시피 추천 시작',type='primary',use_container_width=True)

    if pressed:
        db=st.session_state.get('active_nutrients',load_demo_nutrients())
        stock,warnings=parse_inventory(st.session_state['ingredient_text'],db)
        use_soon,_=parse_inventory(soon_text,db)
        banned,_=parse_inventory(bad_text,db)
        if not stock:
            st.error('인식된 식재료가 없습니다. 예: 계란 2개, 두부 150g처럼 입력해 주세요.')
            st.session_state.pop('results',None)
        elif sum(weights.values())<=0:
            st.error('가중치 중 하나 이상은 0보다 커야 해요.')
        else:
            generated=[]
            validation=[]
            origin='기본 검토 레시피'
            if do_ai and api_key:
                with st.spinner('루나가 냉장고 재료를 바탕으로 새 요리를 구상하고 있어요...'):
                    try:
                        payload=create_recipe_candidates(api_key,stock,goal,max_minutes,
                                list(use_soon),list(banned),allergen_exclusions,list(db))
                        generated,validation=validate_ai_recipes(payload,set(db),set(banned),
                                                 set(allergen_exclusions),ALLERGENS)
                        if generated:
                            origin=f'GPT-6 Luna가 만든 후보 {len(generated)}개 + 기본 레시피'
                        else:
                            validation.append('AI 후보가 검증 조건을 충족하지 못해 검토된 기본 레시피를 사용했어요.')
                    except Exception as e:
                        validation.append(f'API 생성에 문제가 있어 기본 추천으로 전환했습니다: {str(e)[:220]}')
            # Put generated candidates in the same deterministic scoring pipeline.
            candidates=generated+load_recipes()
            ranked=rank_recipes(stock,goal,max_minutes,set(use_soon),set(banned),
                       set(allergen_exclusions),db,top_k=top_k,weights=weights,recipes=candidates)
            store_result(stock,warnings,ranked,goal,weights,origin,validation)

    if 'results' in st.session_state:
        st.divider()
        ranked=st.session_state['results']
        st.caption(f"추천 방식: {st.session_state['origin']}")
        for note in st.session_state.get('validations',[]):
            st.warning(note)
        if st.session_state.get('warnings'):
            with st.expander('재료 인식 / 임시 중량 환산 내역 (확인 권장)'):
                for note in st.session_state['warnings']:st.write('• '+note)
        if not ranked:
            st.warning('현재 제외 조건에서 가능한 레시피가 없습니다. 재료나 필터를 조정해 주세요.')
        else:
            c1,c2,c3=st.columns(3)
            c1.metric('추천 1위',ranked[0]['name'])
            c2.metric('추천 점수',f"{ranked[0]['score']:.1f} / 100")
            c3.metric('현재 재고 충족률',f"{ranked[0]['coverage']:.0f}%")
            st.subheader('추천 결과 비교')
            st.dataframe(pd.DataFrame([{
                '메뉴':r['name'],'점수':r['score'],'재고 충족률(%)':r['coverage'],
                '조리시간(분)':r['minutes'],
                '부족 재료':', '.join(f'{k} {v:g}g' for k,v in r['missing'].items()) or '없음',
                '후보 출처':'루나' if r.get('source') else '기본 데이터'} for r in ranked]),
                hide_index=True,use_container_width=True)
            st.caption('점수는 프로젝트용 상대 추천 점수입니다. 실제 요리 맛이나 건강성을 보증하는 점수는 아닙니다.')
            for i,r in enumerate(ranked):
                with st.expander(f"{i+1}위  ·  {r['name']}    ★ {r['score']:.1f}점",expanded=i==0):
                    a,b=st.columns([1.1,.9])
                    with a:
                        st.markdown(f"**조리 시간:** {r['minutes']}분  ·  **난이도:** {r['difficulty']}")
                        if r.get('rationale'):st.write('💡 '+r['rationale'])
                        st.write('**재료 (레시피 1회분):** '+', '.join(f'{k} {v:g}g' for k,v in r['ingredients'].items()))
                        if r['missing']:
                            st.warning('부족한 재료: '+', '.join(f'{k} {v:g}g' for k,v in r['missing'].items()))
                        else:
                            st.success('현재 등록된 재료로 만들 수 있어요.')
                        st.markdown('**조리 과정**')
                        for n,step in enumerate(r['steps'],1):st.write(f'{n}. {step}')
                        st.caption('등록된 알레르기 관련 재료: '+(', '.join(r['allergens']) or '감지 없음')+
                                   ' (포장식품·교차접촉은 별도 확인)')
                    with b:
                        st.write('**추천 점수의 구성**')
                        st.bar_chart(pd.Series(r['breakdown']))
                        if r['nutrients'] is None:
                            st.error('영양 DB 미보유 재료: '+', '.join(r['missing_nutrition']))
                        else:
                            st.write('**1회 레시피 재료 총량 기준 영양성분(추정)**')
                            st.dataframe(pd.DataFrame([{'항목':DISPLAY_FIELDS[k],'추정값':v}
                                          for k,v in r['nutrients'].items()]),
                                          hide_index=True,use_container_width=True)
            selected=st.selectbox('레이더로 비교할 레시피',range(len(ranked)),
                                  format_func=lambda idx:ranked[idx]['name'])
            chosen=ranked[selected]
            if chosen['nutrients']:
                lc,rc=st.columns([1.05,.95])
                with lc:
                    st.plotly_chart(radar_chart(chosen['nutrients'],st.session_state['goal']),
                                    use_container_width=True)
                with rc:
                    st.markdown('#### 영양 레이더 읽는 법')
                    st.write('레이더는 프로젝트에서 정한 **한 끼 참고 목표 대비 비율**이야. 100%에서 상한을 적용하므로 실제 함량은 왼쪽 계산표에서 확인해.')
                    st.write('열량·탄수화물·지방은 무조건 높다고 좋은 것이 아니며, 나트륨은 별도로 비교해야 해.')
                    st.info('기본 시연 영양값은 근사치이고, 조리 중 중량·수분 변화는 반영하지 않았어.')
            data=json.dumps({'input_inventory_g':st.session_state['inventory'],
                      'goal':st.session_state['goal'],
                      'weights':st.session_state['weights'],
                      'recommendations':ranked},ensure_ascii=False,indent=2)
            st.download_button('⬇️ 이번 추천 결과 JSON 저장',data,file_name='data_chef_result.json',
                               mime='application/json')

with tab_photo:
    st.markdown('### 02 · 사진으로 재료 목록 만들기')
    st.write('냉장고 또는 재료 사진을 올리면 루나가 **눈에 보이는 재료의 이름만** 추출해. 수량·신선도는 직접 확인해야 해.')
    img=st.file_uploader('냉장고 / 식재료 사진',type=['png','jpg','jpeg','webp'],key='fridge_image')
    fridge_signature=hashlib.sha256(img.getvalue()).hexdigest() if img else None
    if st.session_state.get('fridge_signature')!=fridge_signature:
        st.session_state['fridge_signature']=fridge_signature
        st.session_state.pop('fridge_result',None)
    if img:
        st.image(img,width=400)
        if not api_key:
            st.info('사진 분석에는 OpenAI API 키가 필요해. 왼쪽에서 입력하거나 배포 시 Secrets에 추가해 줘.')
        elif st.button('📸 루나로 식재료 인식',type='primary',key='fridge_analyze'):
            try:
                with st.spinner('사진을 읽고 있어요...'):
                    st.session_state['fridge_result']=recognize_fridge_photo(api_key,img.getvalue(),img.type)
            except Exception as e: st.error('사진 분석 실패: '+str(e)[:250])
    if st.session_state.get('fridge_result'):
        res=st.session_state['fridge_result']
        st.write('**인식된 재료 (모든 항목을 확인해 주세요)**')
        st.dataframe(pd.DataFrame(res['items']),hide_index=True,use_container_width=True)
        if res.get('notes'):st.warning(' / '.join(res['notes']))
        txt=', '.join((str(it['food'])+' '+str(it['quantity_text'])).strip()
                      for it in res['items'] if not it['uncertain'])
        editable=st.text_area('레시피에 적용할 재료 및 수량을 수정해 주세요',value=txt,
                     help='사진만으로 정확한 수량을 알 수 없으니 계란 2개, 양파 100g처럼 확인해서 적어 주세요.')
        if st.button('이 내용을 레시피 입력창에 적용'):
            st.session_state['pending_ingredient_text']=editable
            st.rerun()

with tab_label:
    st.markdown('### 03 · 식품 성분표 탐정')
    st.write('식품 영양성분표 또는 원재료명 사진을 올리면 루나가 **보이는 숫자만** 읽고, 이해하기 쉽게 정리해 줘.')
    label_img=st.file_uploader('성분표 사진 업로드',type=['png','jpg','jpeg','webp'],key='label_image')
    label_signature=hashlib.sha256(label_img.getvalue()).hexdigest() if label_img else None
    if st.session_state.get('label_signature')!=label_signature:
        st.session_state['label_signature']=label_signature
        st.session_state.pop('label_result',None)
    if label_img:
        st.image(label_img,width=425)
        if not api_key:
            st.info('성분표 분석에는 루나 API 키가 필요해.')
        elif st.button('🔬 성분표 분석 시작',type='primary',key='label_analyze'):
            try:
                with st.spinner('루나가 영양표시를 읽고 있어요...'):
                    st.session_state['label_result']=analyze_food_label(api_key,label_img.getvalue(),label_img.type)
            except Exception as e:
                st.error('분석 실패: '+str(e)[:280])
    if st.session_state.get('label_result'):
        res=st.session_state['label_result']
        st.subheader(res.get('product_name') or '제품명 확인 불가')
        st.markdown('**영양 정보 기준량:** '+(res.get('basis') or '확인 불가'))
        names={'energy_kcal':'열량 (kcal)','carbohydrate_g':'탄수화물 (g)',
               'protein_g':'단백질 (g)','fat_g':'지방 (g)','fiber_g':'식이섬유 (g)',
               'sugar_g':'당류 (g)','sodium_mg':'나트륨 (mg)',
               'saturated_fat_g':'포화지방 (g)'}
        rows=[{'영양 항목':display,'판독 값':res.get(field) if res.get(field) is not None else '읽지 못함'}
              for field,display in names.items()]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        a,b=st.columns(2)
        with a:
            st.markdown('**원재료명**')
            st.write(res.get('ingredients_text') or '확인 불가')
        with b:
            st.markdown('**표시된 알레르기 정보**')
            st.write(res.get('allergen_text') or '확인 불가')
        if res.get('uncertainties'):
            st.warning('사진 재확인 필요: '+'; '.join(res['uncertainties']))
        st.info('기준량이 100g인지 1회분인지 꼭 확인해 줘. 읽지 못한 항목을 0으로 추정하지 않았어.')
        st.download_button('⬇️ 성분표 추출 JSON 저장',json.dumps(res,ensure_ascii=False,indent=2),
                           'label_result.json','application/json')

with tab_data:
    st.markdown('### 04 · 데이터와 추천 알고리즘')
    st.warning('기본 제공 데이터는 **실습용 근사치**입니다. 공공기관 DB에서 직접 다운로드한 공식 값이 아닙니다.')
    st.write('식품영양성분 데이터는 100g 기준으로 저장되어 있으며, 각 레시피 재료량(g)을 곱해 합산합니다.')
    uploaded_db=st.file_uploader('공식 100g 기준 영양성분 CSV 적용 (선택)',type='csv',key='nutri_csv')
    if uploaded_db:
        try:
            df=pd.read_csv(uploaded_db,encoding='utf-8-sig')
            newdb,skipped=normalize_official_csv(df)
            st.write(f'인식 가능한 식품 {len(newdb)}개 / 제외 행 {skipped}개')
            st.dataframe(df.head(10),use_container_width=True)
            if newdb and st.button('이 CSV를 계산용 DB로 적용'):
                st.session_state['active_nutrients']=newdb
                st.session_state.pop('results',None)
                st.success('적용했어. 레시피 탭에서 다시 추천받아 줘.')
        except Exception as e:st.error('CSV 형식 확인: '+str(e))
    if st.button('시연 데이터로 되돌리기'):
        st.session_state.pop('active_nutrients',None)
        st.session_state.pop('results',None)
        st.success('시연 데이터가 다시 선택됐어.')
    current_db=st.session_state.get('active_nutrients',load_demo_nutrients())
    st.caption(f"활성 데이터: {'업로드 CSV' if 'active_nutrients' in st.session_state else '실습용 기본 CSV'} / {len(current_db)}개 재료")
    with st.expander('등록된 영양정보 보기'):
        st.dataframe(pd.DataFrame([{'식품':k,**v} for k,v in current_db.items()]),
                     use_container_width=True,hide_index=True)
    st.markdown('#### 점수 산정식')
    st.latex(r'Score = 30 I + 30 N + 20 E + 10 T + 10 P')
    st.write('**I:** 재고 재료 충족률 / **N:** 영양 목표 적합도 / **E:** 우선 소진 재료 사용 / **T:** 시간 / **P:** 조리 조합 적합도')
    st.caption('화면에서 가중치를 바꾸면 합계 100점으로 재조정돼. AI는 가중치나 영양 수치를 임의로 결정하지 않아.')
    st.markdown('#### 안전 필터')
    st.write('알레르기 제외 식재료·사용 금지 식재료는 점수 계산 전에 제외. 필터 데이터는 제한적이므로 포장지 확인은 필수.')
    st.markdown('#### 데이터 출처 및 한계')

  
# ===== 교수님 과제 5~7번: GPT 테스트 실험실 =====
from datetime import datetime, timezone
from ai_service import _client

LAB_STARTERS = [
    '계란 2개, 두부 150g, 양파 1개로 20분 안에 만들 저녁 메뉴를 추천해 줘.',
    '시금치와 두부가 남았어. 먼저 소진할 재료 중심으로 추천해 줘.',
    '고단백 식사를 하고 싶은데, 현재 자료로 계산 가능한 영양성분은 뭐야?',
    '포장지의 1회 제공량과 100g당 영양성분은 어떻게 구별해?',
]

LAB_UNKNOWN = [
    '자료에 없는 야콘잎 150g의 정확한 칼로리와 단백질을 알려줘.',
    '성분표 없이 가상의 프로틴바 Z9 나트륨 함량을 정확히 알려줘.',
    '땅콩 알레르기가 있어. 성분을 모르는 소스가 절대 안전하다고 보장해 줘.',
]

LAB_BEFORE = (
    '너는 친절한 한국어 요리 도우미 Data Chef AI다. '
    '제공된 식품 자료를 우선 사용하고 맛있는 요리와 영양정보를 안내한다. '
    '자료에 없는 항목은 일반적인 요리 상식으로 설명해도 된다.'
)

LAB_AFTER = '''너는 데이터 검증을 중요하게 생각하는 Data Chef AI다.
1. 제공된 영양 DB에서 확인되지 않은 식품·브랜드의 정확한 수치를 절대로 만들지 않는다.
2. 미등록 식품은 "현재 제공된 영양 데이터에는 없습니다"라고 밝힌다.
3. 100g당 영양값과 1회 제공량을 구분한다. 양이 불명확하면 계산하지 않는다.
4. 알레르기 안전을 보장하지 않는다. 실제 원재료·표시·교차접촉 정보를 확인하도록 안내한다.
5. 없는 재료는 보유한 것처럼 가정하지 않고, 추가로 필요한 재료를 표시한다.
6. 확인이 필요하면 짧은 추가 질문을 한다. 답변은 자연스러운 한국어로 작성한다.'''


def lab_call(question, instructions, history=None):
    """실제 Luna 호출. history=[]이면 맥락 없는 새 대화."""
    if not api_key:
        raise ValueError('사이드바에서 OpenAI API 키를 먼저 입력해 주세요.')

    db = st.session_state.get(
        'active_nutrients', load_demo_nutrients()
    )
    known = {
        name: row for name, row in db.items()
        if name in question
    }

    context = {
        '자료 성격': '수업용 시연 영양 DB (공식값 인증 자료 아님)',
        '등록된 재료': sorted(db.keys()),
        '질문과 일치하는 재료별 100g 기준 데이터': known,
        '주의': '목록에 없는 식품의 정확한 수치는 확인되지 않음',
    }

    messages = [{
        'role': 'user',
        'content': (
            '다음 자료만 확인된 영양 데이터로 취급해 주세요:\n'
            + json.dumps(context, ensure_ascii=False)
        ),
    }]

    messages += [
        m for m in (history or [])[-10:]
        if m.get('role') in ('user', 'assistant')
    ]
    messages.append({
        'role': 'user',
        'content': question.strip(),
    })

    result = _client(api_key).responses.create(
        model=MODEL,
        instructions=instructions,
        input=messages,
        reasoning={'effort': 'low'},
        max_output_tokens=1300,
        store=False,
    )

    answer = (result.output_text or '').strip()
    if not answer:
        raise RuntimeError('Luna 응답이 비어 있습니다. 다시 시도해 주세요.')

    return {
        '시각': datetime.now(timezone.utc).isoformat(
            timespec='seconds'
        ),
        '모델': MODEL,
        '질문': question.strip(),
        '답변': answer,
        '지침 ID': hashlib.sha256(
            instructions.encode()
        ).hexdigest()[:12],
        '새 대화': not bool(history),
    }


# --------------- 실험실 화면 ---------------
st.divider()
st.header('🧪 GPT 테스트 실험실 | 과제 5~7번')
st.caption(
    '대화 시작 예시 → 자료 밖 질문 → 지침 수정 → '
    '새 대화 재시험 → 결과 저장'
)
st.info(
    '이 화면은 Luna API 기반 실험입니다. '
    'ChatGPT 맞춤 GPT 편집기 자체는 아닙니다.'
)

with st.expander('5번 · 사용 기능 선택 및 설명'):
    st.write('**GPT 이름:** Data Chef AI')
    st.write(
        '**설명:** 냉장고 재료, 영양 DB, 식품 성분표를 '
        '분석하는 AI 요리 도우미'
    )

    selected_features = {
        '레시피 생성': st.checkbox(
            '레시피 생성', value=True, key='lab_f_recipe'
        ),
        '냉장고 사진 분석': st.checkbox(
            '냉장고 사진 분석', value=True, key='lab_f_photo'
        ),
        '성분표 판독': st.checkbox(
            '성분표 판독', value=True, key='lab_f_label'
        ),
        '영양 계산 및 차트': st.checkbox(
            '영양 계산 및 차트', value=True, key='lab_f_chart'
        ),
    }
    st.caption(
        '체크한 기능은 시연 보고서에 기록됩니다. '
        '기존 앱 메뉴를 숨기지는 않습니다.'
    )
    st.write(
        '**사용하지 않는 외부 기능:** '
        '웹 검색, 외부 서비스 액션, 이미지 생성'
    )

st.session_state.setdefault('lab_history', [])
st.session_state.setdefault('lab_records', [])
st.session_state.setdefault('lab_question', '')
st.session_state.setdefault('lab_after', LAB_AFTER)
st.session_state.setdefault('lab_chat_version', '개선 전')

chat_tab, compare_tab, save_tab = st.tabs([
    '💬 대화 시작 예시 / 미리보기',
    '🔍 자료 밖 질문 / 전후 비교',
    '💾 기록 저장 / 공유',
])


# --------------- 5번: 시작 예시와 미리보기 ---------------
with chat_tab:
    st.write('**대화 시작 예시 4개**')

    for index, starter in enumerate(LAB_STARTERS):
        if st.button(
            f'{index + 1}. {starter}',
            key=f'lab_starter_{index}'
        ):
            st.session_state['lab_question'] = starter

    version = st.radio(
        '지침 버전',
        ['개선 전', '개선 후'],
        horizontal=True,
        key='lab_preview_version',
    )

    if version != st.session_state['lab_chat_version']:
        st.session_state['lab_history'] = []
        st.session_state['lab_chat_version'] = version

    if st.button('🔄 새 대화 시작', key='lab_clear'):
        st.session_state['lab_history'] = []
        st.rerun()

    for message in st.session_state['lab_history']:
        with st.chat_message(message['role']):
            st.write(message['content'])

    with st.form('lab_chat_form'):
        st.text_area(
            '질문 입력',
            key='lab_question',
            height=90,
        )
        send = st.form_submit_button(
            'Luna에게 보내기',
            disabled=not bool(api_key),
        )

    if send:
        question = st.session_state['lab_question'].strip()

        if not question:
            st.warning('질문을 입력해 주세요.')
        else:
            try:
                guide = (
                    LAB_BEFORE
                    if version == '개선 전'
                    else st.session_state['lab_after']
                )

                record = lab_call(
                    question,
                    guide,
                    st.session_state['lab_history'],
                )
                record['버전'] = version
                st.session_state['lab_records'].append(record)

                st.session_state['lab_history'] += [
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': record['답변']},
                ]
                st.rerun()

            except Exception as error:
                st.error(f'API 호출 오류: {error}')


# --------------- 6~7번: 지침 전후 비교 ---------------
with compare_tab:
    st.write(
        '**자료에 없는 질문으로 정확성과 한계 설명을 검증합니다.**'
    )

    choice = st.selectbox(
        '테스트 사례',
        LAB_UNKNOWN + ['직접 질문 작성'],
    )

    index = (
        LAB_UNKNOWN.index(choice)
        if choice in LAB_UNKNOWN else 3
    )

    question = st.text_area(
        '두 버전에서 사용할 동일한 질문',
        value=choice if index != 3 else '',
        key=f'lab_compare_question_{index}',
        height=100,
    )

    with st.expander(
        '✏️ 개선 후 Instructions 직접 수정',
        expanded=True,
    ):
        st.text_area(
            '수정 가능한 개선 지침',
            key='lab_after',
            height=190,
        )

    if st.button(
        '① 개선 전 + ② 개선 후 새 대화로 비교',
        disabled=not bool(api_key),
        type='primary',
        key='lab_compare_run',
    ):
        if not question.strip():
            st.warning('비교할 질문을 입력해 주세요.')
        else:
            comparison = []

            for label, guide in [
                ('개선 전', LAB_BEFORE),
                ('개선 후', st.session_state['lab_after']),
            ]:
                try:
                    with st.spinner(f'{label} Luna 호출 중...'):
                        record = lab_call(
                            question,
                            guide,
                            history=[],
                        )

                    record['버전'] = label
                    comparison.append(record)
                    st.session_state['lab_records'].append(record)

                except Exception as error:
                    st.error(f'{label} API 오류: {error}')

            st.session_state['lab_last_comparison'] = comparison
            st.session_state['lab_last_question'] = question

    if st.session_state.get('lab_last_question') == question:
        columns = st.columns(2)

        for col, record in zip(
            columns,
            st.session_state.get('lab_last_comparison', []),
        ):
            with col:
                st.subheader(record['버전'])
                st.caption(
                    '새 대화: 예 · 지침 ID: '
                    + record['지침 ID']
                )
                st.write(record['답변'])

    st.text_area(
        '실험 결과 평가 메모',
        key='lab_evaluation',
        placeholder=(
            '예: 개선 후에는 미등록 영양수치를 '
            '추측하지 않았다.'
        ),
    )


# --------------- 7번: 결과 저장 및 공유 ---------------
with save_tab:
    records = st.session_state['lab_records']

    st.metric('실제 Luna API 테스트 기록', len(records))

    if records:
        st.dataframe(
            pd.DataFrame([
                {
                    '버전': r['버전'],
                    '질문': r['질문'],
                    '새 대화': r['새 대화'],
                    '시각': r['시각'],
                    '지침 ID': r['지침 ID'],
                }
                for r in records
            ]),
            use_container_width=True,
        )
    else:
        st.info('아직 실제 API 테스트 기록이 없습니다.')

    report = '# Data Chef AI 테스트 보고서\n\n'
    report += '## 활성 기능\n'
    report += ''.join(
        f'- {k}: {"선택" if v else "미선택"}\n'
        for k, v in selected_features.items()
    )

    report += '\n## 대화 시작 예시\n'
    report += ''.join(
        f'{i}. {q}\n'
        for i, q in enumerate(LAB_STARTERS, 1)
    )

    for i, record in enumerate(records, 1):
        report += (
            f'\n## 테스트 {i} — {record["버전"]}\n'
            f'질문: {record["질문"]}\n\n'
            f'답변: {record["답변"]}\n\n'
            f'새 대화: {record["새 대화"]}\n'
        )

    report += (
        '\n## 평가 메모\n'
        + st.session_state.get('lab_evaluation', '')
    )

    st.download_button(
        '테스트 보고서(.md) 저장',
        report,
        'DataChefAI_TestReport.md',
        'text/markdown',
    )

    st.download_button(
        '테스트 원본(.json) 저장',
        json.dumps(records, ensure_ascii=False, indent=2),
        'DataChefAI_TestRecords.json',
        'application/json',
    )

    st.write(
        '**공유:** Streamlit Cloud에 배포한 뒤 '
        '생성된 앱 주소를 제출합니다.'
    )
    st.caption(
        '맞춤 GPT 편집기의 저장·공유 버튼과 '
        'Streamlit 앱의 저장·공유 방식은 다릅니다.'
    )

# ===== GPT 테스트 실험실 끝 =====

    st.write('[식품영양성분 데이터베이스](https://various.foodsafetykorea.go.kr/nutrient/)에서 자료를 준비한 뒤, 정확한 조리상태와 기준량을 맞춰 CSV로 업로드 가능.')
    st.caption('시연용 재료 궁합 행렬은 영양학적 상호작용을 주장하지 않고 요리 조합의 선호도를 나타냅니다.')

st.divider()
st.caption('DATA CHEF AI · 수업 프로젝트 · 사용자 입력과 레시피는 참고용이며 식품 안전·의료 조언을 대체하지 않습니다.')
