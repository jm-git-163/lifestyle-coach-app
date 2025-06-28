import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import openai
import base64
import requests
import re
import av
import io
import json
import os
import pandas as pd
from datetime import datetime, time, timedelta
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- 데이터 저장/불러오기 함수 ---
DATA_FILE = "user_data.json"

def save_data(data):
    """세션 상태의 데이터를 JSON 파일에 저장합니다."""
    data_to_save = {}
    for key, value in data.items():
        if not key.startswith('_') and not callable(value) and 'streamlit' not in str(type(value)):
            if isinstance(value, dict):
                data_to_save[key] = value.copy()
                for med_key, med_val in data_to_save[key].items():
                     if isinstance(med_val, list):
                         for item in med_val:
                             if isinstance(item, dict) and 'time' in item and isinstance(item['time'], time):
                                 item['time'] = item['time'].strftime('%H:%M')
            else:
                data_to_save[key] = value
    
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

def load_data():
    """JSON 파일에서 데이터를 불러옵니다."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'medications' in data:
                    for med in data.get('medications', []):
                        if isinstance(med.get('time'), str):
                            med['time'] = datetime.strptime(med['time'], '%H:%M').time()
                return data
            except json.JSONDecodeError: 
                return None
    return None

# --- 유틸리티 함수 ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def autoplay_audio(audio_bytes: bytes, key: str):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f'<audio controls autoplay="true" style="display:none;" key="{key}"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    st.markdown(md, unsafe_allow_html=True)

def text_to_speech(text: str):
    try:
        tts = gTTS(text=text, lang='ko')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.getvalue()
    except Exception as e:
        st.error(f"음성 변환 중 오류 발생: {e}")
        return None

def get_ai_response(api_key, messages, max_tokens=1024):
    if not api_key: return "오류: OpenAI API 키가 제공되지 않았습니다."
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=max_tokens, temperature=0.7)
        return response.choices[0].message.content
    except Exception as e: return f"API 호출 중 오류가 발생했습니다: {e}"

@st.cache_data(ttl=600)
def get_weather_data():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=35.1796&longitude=129.0756&current=temperature_2m,relative_humidity_2m"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"날씨 정보 조회 중 오류 발생: {e}")
        return None

# --- AI 피트니스 코치 클래스 ---
class PoseAnalyzerTransformer(VideoTransformerBase):
    def __init__(self, mode='squat'):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mode = mode
        self.stage = None
        self.counter = 0
        self.feedback = "자세를 잡아주세요"

    def _analyze_squat(self, landmarks, frame_shape):
        shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]
        knee_angle = calculate_angle(hip, knee, ankle)
        if knee_angle > 160: self.stage = "up"
        if knee_angle < 90 and self.stage == 'up':
            self.stage = "down"; self.counter += 1; self.feedback = f"[{self.counter}] 좋은 자세!"
        if self.stage == 'down':
            if knee[0] * frame_shape[1] > (ankle[0] * frame_shape[1] + 20): self.feedback = f"[{self.counter}] 무릎이 발끝을 넘었어요!"
            else: self.feedback = f"[{self.counter}] 자세를 유지하세요!"
        else: self.feedback = f"COUNT: {self.counter}"
            
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = results.pose_landmarks.landmark
            frame_shape = img.shape
            if self.mode == 'squat': self._analyze_squat(landmarks, frame_shape)
            
            if self.feedback != st.session_state.get('last_fitness_feedback', ''):
                feedback_for_speech = re.sub(r'\[\d+\]\s*|COUNT: \d+', '', self.feedback).strip()
                if feedback_for_speech:
                    audio_bytes = text_to_speech(feedback_for_speech)
                    if audio_bytes: autoplay_audio(audio_bytes, key=f"fitness_fb_{int(time.time())}")
                st.session_state.last_fitness_feedback = self.feedback
        except Exception: self.feedback = "카메라에 전신이 보이도록 서주세요"
        cv2.rectangle(image, (0,0), (640,60), (20, 20, 20), -1)
        cv2.putText(image, self.feedback, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- 페이지 렌더링 함수들 ---
def render_dashboard():
    st.title("✨ 오늘의 종합 브리핑")
    # ... (기존 기능 유지) ...
    current_hour = datetime.now().hour
    if 6 <= current_hour < 12: alert_text = "좋은 아침! 💧 물 한 잔으로 하루를 시작하고, 등록된 약을 챙기세요."
    elif 12 <= current_hour < 18: alert_text = "나른한 오후, 🤸 잠시 스트레칭으로 몸을 깨워주세요."
    elif 18 <= current_hour < 24: alert_text = "하루 마무리 시간! 🧴 스킨케어로 피부에 휴식을 주세요."
    else: alert_text = "편안한 밤을 보내세요."
    st.info(alert_text)

    # 1. 물 마시기 체크 기능 복구
    st.subheader("💧 오늘의 수분 섭취")
    p = st.session_state.profile
    recommended_ml = (p.get('height', 170) + p.get('weight', 65)) * 10
    recommended_glasses = round(recommended_ml / 200)
    water_intake = st.session_state.water_intake
    st.write(f"{p.get('gender', '여성')} {p.get('age', 30)}세의 권장 섭취량은 약 **{recommended_glasses}** 잔 입니다.")
    st.progress(min(1.0, water_intake / recommended_glasses))
    cols = st.columns(10)
    for i in range(recommended_glasses):
        with cols[i % 10]:
            if i < water_intake:
                if st.button("💧", key=f"water_filled_{i}", help="마신 물 취소"):
                    st.session_state.water_intake -= 1; st.rerun()
            else:
                if st.button("💧", key=f"water_empty_{i}", type="secondary", help="물 한 잔 마시기"):
                    st.session_state.water_intake += 1
                    remaining = recommended_glasses - st.session_state.water_intake
                    if remaining > 0: feedback = f"좋아요! 권장량까지 {remaining}잔 남았습니다."
                    elif remaining == 0: feedback = "훌륭해요! 오늘 권장 수분량을 모두 채웠습니다."
                    else: feedback = f"충분히 마셨어요! 권장량보다 {abs(remaining)}잔 더 마셨네요."
                    audio = text_to_speech(feedback)
                    if audio: autoplay_audio(audio, key=f"water_feedback_{st.session_state.water_intake}")
                    st.success(feedback); st.rerun()

    st.divider()
    st.subheader("🧴 데일리 스킨케어 체크")
    today_str = str(datetime.now().date())
    col1, col2 = st.columns(2)
    skincare_items = st.session_state.skincare_items
    with col1:
        with st.expander("🌞 아침 스킨케어", expanded=True):
            for item, korean_name in skincare_items.items():
                st.session_state.skincare_routine['morning'][item] = st.checkbox(korean_name, value=st.session_state.skincare_routine['morning'].get(item, False), key=f"morning_{item}")
    with col2:
        with st.expander("🌙 저녁 스킨케어", expanded=True):
            for item, korean_name in skincare_items.items():
                if item != 'sunscreen':
                    st.session_state.skincare_routine['evening'][item] = st.checkbox(korean_name, value=st.session_state.skincare_routine['evening'].get(item, False), key=f"evening_{item}")
    
    if all(st.session_state.skincare_routine['morning'].values()) and not st.session_state.skincare_log.get(today_str, {}).get('morning_audio', False):
        st.session_state.skincare_log.setdefault(today_str, {})['morning_audio'] = True
        audio = text_to_speech("아침 스킨케어 완료! 오늘도 빛나는 하루 보내세요.")
        if audio: autoplay_audio(audio, "morning_complete")
        st.success("✨ 아침 스킨케어를 모두 완료했어요!")
    evening_tasks = {k: v for k, v in st.session_state.skincare_routine['evening'].items() if k != 'sunscreen'}
    if all(evening_tasks.values()) and not st.session_state.skincare_log.get(today_str, {}).get('evening_audio', False):
        st.session_state.skincare_log.setdefault(today_str, {})['evening_audio'] = True
        audio = text_to_speech("저녁 스킨케어 완료! 편안한 밤 되세요.")
        if audio: autoplay_audio(audio, "evening_complete")
        st.success("✨ 저녁 스킨케어를 모두 완료했어요!")
    st.session_state.skincare_log.setdefault(today_str, {})['morning'] = all(st.session_state.skincare_routine['morning'].values())
    st.session_state.skincare_log.setdefault(today_str, {})['evening'] = all(evening_tasks.values())
    st.subheader("🗓️ 이번 주 스킨케어 달성 기록")
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    days = [(week_start + timedelta(days=i)) for i in range(7)]
    cols = st.columns(7)
    for i, day in enumerate(days):
        with cols[i]:
            day_str = str(day.date())
            st.markdown(f"<p style='text-align: center;'>{day.strftime('%a')}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 24px;'>{day.day}</p>", unsafe_allow_html=True)
            log = st.session_state.skincare_log.get(day_str, {})
            morning_check = "✅" if log.get('morning') else "➖"
            evening_check = "✅" if log.get('evening') else "➖"
            st.markdown(f"<p style='text-align: center;'>🌞 {morning_check} 🌙 {evening_check}</p>", unsafe_allow_html=True)

def render_profile(api_key):
    st.title("👤 내 프로필 & 목표 설정")
    p = st.session_state.profile
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("신체 정보")
        p['gender'] = st.radio("성별", ["여성", "남성"], index=["여성", "남성"].index(p.get('gender', '여성')), horizontal=True)
        p['age'] = st.number_input("나이", value=p.get('age', 30), min_value=1, max_value=120)
        p['height'] = st.number_input("키 (cm)", value=p.get('height', 170.0), format="%.1f")
        p['weight'] = st.number_input("현재 체중 (kg)", value=p.get('weight', 65.0), format="%.1f")
    with col2:
        st.subheader("목표 설정")
        p['goal_weight'] = st.number_input("목표 체중 (kg)", value=p.get('goal_weight', 60.0), format="%.1f")
        if st.button("🎯 AI 목표 달성 플랜 받기", use_container_width=True, type="primary"):
            if not api_key: st.warning("AI 플랜 기능을 사용하려면 사이드바에 OpenAI API 키를 입력해주세요.")
            else:
                profile_info = f"사용자 정보: {p.get('age')}세 {p.get('gender')}, 키 {p.get('height')}cm, 현재 체중 {p.get('weight')}kg, 목표 체중 {p.get('goal_weight')}kg"
                prompt = f"{profile_info} 상태입니다. 이 사용자의 목표 달성을 위한 주간 식단 및 운동 계획을 구체적으로 제안해주세요. 칭찬과 격려를 담아 친근한 말투로 작성해주세요."
                with st.spinner("AI가 당신만을 위한 플랜을 짜는 중입니다..."):
                    response = get_ai_response(api_key, [{"role": "user", "content": prompt}])
                    st.session_state.ai_plan = response
    
    if 'ai_plan' in st.session_state:
        with st.expander("🤖 AI가 제안하는 맞춤 플랜", expanded=True): st.markdown(st.session_state.ai_plan)
    st.divider()
    st.subheader("⚖️ 체중 변화 기록")
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.form("weight_log_form", clear_on_submit=True):
            log_date = st.date_input("기록일", datetime.now())
            log_weight = st.number_input("기록할 체중 (kg)", min_value=0.0, value=p['weight'], format="%.1f")
            if st.form_submit_button("기록하기"):
                found = False
                for record in st.session_state.weight_log:
                    if record['date'] == str(log_date): record['weight'] = log_weight; found = True; break
                if not found: st.session_state.weight_log.append({'date': str(log_date), 'weight': log_weight})
                st.success(f"{log_date}의 체중 {log_weight}kg이 기록되었습니다.")
    with col2:
        if st.session_state.weight_log:
            log_df = pd.DataFrame(st.session_state.weight_log)
            log_df['date'] = pd.to_datetime(log_df['date'])
            log_df = log_df.sort_values(by='date').set_index('date')
            st.line_chart(log_df)
        else: st.info("체중을 기록하여 변화를 확인해보세요.")
    st.divider()
    st.subheader("💊 복용 약 관리 및 알림")
    with st.form("med_form", clear_on_submit=True):
        med_name = st.text_input("약 이름")
        med_days = st.multiselect("복용 요일", options=['월', '화', '수', '목', '금', '토', '일'], default=['월', '화', '수', '목', '금', '토', '일'])
        med_time_obj = st.time_input("복용 시간", value=time(8, 30))
        if st.form_submit_button("추가하기") and med_name and med_days:
            st.session_state.medications.append({'name': med_name, 'time': med_time_obj, 'days': med_days})
            st.success(f"'{med_name}' 약이 추가되었습니다.")
    
    if st.session_state.medications:
        st.write("등록된 약 목록:")
        for i, med in enumerate(st.session_state.medications):
            col1, col2 = st.columns([4,1])
            col1.write(f"- {med['name']} ({', '.join(med.get('days',[]))}, {med.get('time').strftime('%H:%M')})")
            if col2.button("삭제", key=f"del_med_{i}"): st.session_state.medications.pop(i); st.rerun()

def render_inventory(api_key):
    st.title("🛒 쇼핑 & 인벤토리")
    # 2. 사진 업로드 기능 복구
    tab1, tab2 = st.tabs(["🛍️ 쇼핑하기", "📷 AI 스캔으로 직접 추가"])
    with tab1:
        st.subheader("🥦 신선 식품 코너")
        fresh_food = { "닭가슴살": {'name': '닭가슴살', 'type': 'food', 'tags': ['고단백', '저지방']}, "계란 (30구)": {'name': '계란', 'type': 'food', 'tags': ['고단백', '완전식품']}, "두부": {'name': '두부', 'type': 'food', 'tags': ['식물성단백질', '건강식']}, "브로콜리": {'name': '브로콜리', 'type': 'food', 'tags': ['채소', '비타민']}, "아보카도": {'name': '아보카도', 'type': 'food', 'tags': ['건강한지방']}, "연어": {'name': '연어', 'type': 'food', 'tags': ['오메가3']}, "고구마": {'name': '고구마', 'type': 'food', 'tags': ['탄수화물', '다이어트']}, "퀴노아": {'name': '퀴노아', 'type': 'food', 'tags': ['슈퍼푸드']} }
        cols = st.columns(4)
        for i, (name, data) in enumerate(fresh_food.items()):
            cols[i%4].button(name, key=f"fresh_{name}", on_click=lambda d=data: st.session_state.cart.append(d))
        st.subheader("🍜 가공/간편 식품 코너")
        processed_food = { "초코 케이크": {'name': '초코 케이크', 'type': 'food', 'tags': ['고칼로리', '디저트', '주의필요']}, "신라면 (5개입)": {'name': '라면', 'type': 'food', 'tags': ['고나트륨', '인스턴트', '주의필요']}, "냉동피자": {'name': '냉동피자', 'type': 'food', 'tags': ['고칼로리', '인스턴트', '주의필요']}, "감자칩": {'name': '감자칩', 'type': 'food', 'tags': ['고나트륨', '과자', '주의필요']}, "프로틴바": {'name': '프로틴바', 'type': 'food', 'tags': ['단백질보충']}, "제로콜라": {'name': '제로콜라', 'type': 'food', 'tags': ['음료']} }
        cols = st.columns(4)
        for i, (name, data) in enumerate(processed_food.items()):
            cols[i%4].button(name, key=f"proc_{name}", on_click=lambda d=data: st.session_state.cart.append(d))
        st.subheader("🧴 스킨케어 코너")
        cosmetics = { "수분 크림": {'name': '수분 크림', 'type': 'cosmetic', 'tags': ['보습', '데일리', '크림']}, "티트리 마스크": {'name': '티트리 마스크', 'type': 'cosmetic', 'tags': ['진정', '트러블케어', '마스크팩']}, "클렌징 폼": {'name': '클렌징 폼', 'type': 'cosmetic', 'tags': ['세안', '필수', '클렌저']}, "토너/스킨": {'name': '토너/스킨', 'type': 'cosmetic', 'tags': ['피부결정돈', '기초', '토너']}, "에센스/세럼": {'name': '에센스/세럼', 'type': 'cosmetic', 'tags': ['영양공급', '기능성', '세럼']}, "선크림": {'name': '선크림', 'type': 'cosmetic', 'tags': ['자외선차단', '필수', '선크림']}, "알로에 젤": {'name': '알로에 젤', 'type': 'cosmetic', 'tags': ['진정', '수분공급', '젤']}, "아이 크림": {'name': '아이 크림', 'type': 'cosmetic', 'tags': ['눈가관리', '주름개선']}, "클렌징 오일": {'name': '클렌징 오일', 'type': 'cosmetic', 'tags': ['메이크업제거']}, "보습 로션": {'name': '보습 로션', 'type': 'cosmetic', 'tags': ['보습', '데일리', '로션']} }
        cols = st.columns(4)
        for i, (name, data) in enumerate(cosmetics.items()):
            cols[i%4].button(name, key=f"cosm_{name}", on_click=lambda d=data: st.session_state.cart.append(d))
    
    with tab2:
        st.info("쇼핑하지 않고 직접 구매한 상품의 사진을 찍어 인벤토리에 바로 추가할 수 있습니다.")
        uploaded_file = st.file_uploader("상품 사진 업로드", type=["jpg", "png", "jpeg"], key="direct_add")
        if uploaded_file and api_key:
            st.image(uploaded_file)
            if st.button("AI로 분석 후 인벤토리에 추가", use_container_width=True):
                with st.spinner("AI가 상품을 분석 중입니다..."):
                    image_bytes = uploaded_file.getvalue()
                    b64_image = base64.b64encode(image_bytes).decode('utf-8')
                    prompt = "이 이미지 속 상품이 '음식'인지 '화장품'인지 먼저 판단하고, 상품의 이름과 특징을 분석해서 JSON 형식으로 알려줘. 형식: {\"name\": \"상품이름\", \"type\": \"food/cosmetic\", \"tags\": [\"태그1\", \"태그2\"]}"
                    response = get_ai_response(api_key, [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]}])
                    try:
                        json_match = re.search(r'```json\n({.*?})\n```', response, re.DOTALL)
                        if json_match:
                            item_data_str = json_match.group(1)
                            item_data = json.loads(item_data_str)
                            if item_data.get('type') == 'food': 
                                st.session_state.food_inventory.append(item_data)
                            elif item_data.get('type') == 'cosmetic': 
                                st.session_state.cosmetic_inventory.append(item_data)
                            st.success(f"'{item_data.get('name')}'을(를) 인벤토리에 추가했습니다!")
                            # 스캔 추가 시에도 경고 기능 작동
                            if '주의필요' in item_data.get('tags', []):
                                st.warning(f"**주의!** '{item_data.get('name')}'은(는) 목표 체중 달성에 방해가 될 수 있습니다. 건강한 식단을 유지해 주세요!")
                        else:
                            st.error("AI 분석 결과에서 JSON 데이터를 찾을 수 없습니다.")
                    except Exception as e: st.error(f"AI 분석 결과를 처리하는 데 실패했습니다: {e}")
        elif uploaded_file and not api_key:
            st.warning("AI 분석 기능을 사용하려면 사이드바에 API 키를 입력해주세요.")

    st.divider()
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🛒 내 장바구니")
        if not st.session_state.cart: st.write("장바구니가 비었습니다.")
        else:
            for item in st.session_state.cart: st.write(f"- {item['name']}")
            # 3. 쇼핑 경고 기능 수정
            if st.button("결제하기", use_container_width=True, type="primary"):
                caution_items = []
                for item in st.session_state.cart:
                    if item['type'] == 'food': st.session_state.food_inventory.append(item)
                    elif item['type'] == 'cosmetic': st.session_state.cosmetic_inventory.append(item)
                    if '주의필요' in item.get('tags', []): caution_items.append(item['name'])
                st.session_state.cart = []
                st.success("결제 완료! 인벤토리에 상품이 추가되었습니다.")
                if caution_items:
                    st.warning(f"**주의!** '{', '.join(caution_items)}'은(는) 목표 체중 달성에 방해가 될 수 있습니다. 건강한 식단을 유지해 주세요!")
                st.rerun()
    with col2:
        st.subheader("🧊 내 인벤토리")
        with st.container(height=300):
            st.write("**🍎 식품 냉장고**")
            if not st.session_state.food_inventory: st.write("비어있음")
            else: [st.write(f"- {item['name']} `[{', '.join(item.get('tags', []))}]`") for item in st.session_state.food_inventory]
            st.write("**🧴 화장품 냉장고**")
            if not st.session_state.cosmetic_inventory: st.write("비어있음")
            else: [st.write(f"- {item['name']} `[{', '.join(item.get('tags', []))}]`") for item in st.session_state.cosmetic_inventory]

def get_season():
    month = datetime.now().month
    if month in [3, 4, 5]: return "봄"
    if month in [6, 7, 8]: return "여름"
    if month in [9, 10, 11]: return "가을"
    return "겨울"

def render_recommendations(api_key):
    st.title("💡 AI 라이프스타일 추천")
    if not api_key: st.warning("AI 추천 기능을 사용하려면 사이드바에 OpenAI API 키를 입력해주세요."); st.stop()
    weather_data = get_weather_data()
    if not weather_data: st.error("날씨 정보를 불러올 수 없습니다."); st.stop()
    temp = weather_data['current']['temperature_2m']
    humidity = weather_data['current']['relative_humidity_2m']
    st.subheader(f"현재 부산 날씨: {temp}°C, 습도 {humidity}%")
    
    tab1, tab2, tab3 = st.tabs(["👗 스타일링 & 스킨케어", "🥗 레시피", "🚗 여행지"])
    with tab1:
        st.subheader("👕 날씨 맞춤 코디 추천")
        if temp >= 28: outfit = "시원한 반팔과 반바지"
        elif temp >= 20: outfit = "가벼운 셔츠나 긴팔"
        elif temp >= 12: outfit = "가디건이나 얇은 재킷"
        else: outfit = "따뜻한 코트나 패딩"
        st.info(f"**AI 추천 코디:** 오늘은 **{outfit}** 차림이 좋겠어요.")

        st.subheader("🧴 날씨 맞춤 스킨케어 추천")
        # 4. AI 스킨케어 추천 오류 수정
        if st.button("AI 스킨케어 추천받기", key="skincare_rec", use_container_width=True):
            with st.spinner("AI가 피부와 날씨를 분석중입니다..."):
                prompt = f"현재 부산 날씨는 기온 {temp}°C, 습도 {humidity}% 입니다. 이 날씨에 가장 효과적인 스킨케어 제품 타입 딱 하나만 '제품명: [제품타입]' 형식으로 추천해줘. 예: '제품명: 수분 크림'. 다른 설명은 절대 붙이지마."
                item_name_response = get_ai_response(api_key, [{"role": "user", "content": prompt}])
                st.session_state.skincare_rec_response = item_name_response
        
        if 'skincare_rec_response' in st.session_state:
            response_text = st.session_state.skincare_rec_response
            # AI 응답에서 '제품명: ' 뒷부분만 정확히 추출
            match = re.search(r"제품명:\s*(.+)", response_text)
            if match:
                recommended_item = match.group(1).strip()
                st.info(f"**AI 추천:** {recommended_item}")
                cosmetic_inventory_names = [item['name'] for item in st.session_state.cosmetic_inventory]
                if recommended_item in cosmetic_inventory_names:
                    st.success(f"**진단 결과:** 마침 화장품 냉장고에 '{recommended_item}'이(가) 있네요! 오늘 사용해보세요.")
                else:
                    st.warning(f"**진단 결과:** 아쉽게도 '{recommended_item}'이(가) 인벤토리에 없어요. 구매를 추천합니다.")
            else:
                st.error("AI 추천 제품을 분석하는데 실패했습니다. 다시 시도해주세요.")
            
    with tab2:
        st.subheader("🥗 인벤토리 기반 레시피 추천")
        st.info("AI가 당신의 냉장고 속 재료와 목표를 분석하여 건강 레시피를 제안합니다.")
        if st.button("AI 레시피 추천받기", use_container_width=True):
            healthy_items = [item['name'] for item in st.session_state.food_inventory if '주의필요' not in item.get('tags', [])]
            with st.spinner("AI가 레시피를 구상 중입니다..."):
                profile_info = f"현재 체중 {st.session_state.profile['weight']}kg, 목표 체중 {st.session_state.profile['goal_weight']}kg인 사용자"
                if len(healthy_items) < 2:
                    prompt = f"현재 냉장고 속 건강 재료가 거의 없습니다. {profile_info}를 위한 맛있고 건강한 다이어트 식단을 위해 구매하면 좋을 식재료 5가지를 추천해주세요. 간단한 장보기 목록 형식으로요."
                    st.warning("냉장고 속 건강한 재료가 부족하네요. AI가 장보기 목록을 추천합니다.")
                else:
                    prompt = f"사용자 정보: {profile_info}\n현재 냉장고 재료: {', '.join(healthy_items)}\n계절: {get_season()}\n위 정보를 모두 종합하여, 사용자의 목표 달성에 도움이 될 '건강한 다이어트 레시피' 1가지를 제안해주세요. 레시피 이름, 필요한 재료 목록(만약 현재 냉장고에 없는 추가 재료가 필요하다면 '추가 필요 재료' 항목에 꼭 명시해주세요), 그리고 조리 단계를 순서대로 명확하게 알려주세요."
                response = get_ai_response(api_key, [{"role": "user", "content": prompt}])
                st.session_state.recipe_response = response
        if 'recipe_response' in st.session_state: st.markdown(st.session_state.recipe_response)

    with tab3:
        st.subheader("🚗 근교 여행지 추천")
        if st.button("AI 여행지 추천받기", use_container_width=True):
            prompt = f"현재 위치는 부산이고, 날씨는 {temp}°C, 계절은 {get_season()}입니다. 이 모든 조건에 어울리는 부산 근교의 당일치기 여행지나 최근 뜨는 핫플레이스 3곳을 추천하고, 각 장소에서 즐길 수 있는 활동을 간단히 설명해줘."
            with st.spinner("AI가 여행지를 검색 중입니다..."):
                response = get_ai_response(api_key, [{"role": "user", "content": prompt}])
                st.session_state.trip_response = response
        if 'trip_response' in st.session_state: st.markdown(st.session_state.trip_response)

def render_fitness_coach():
    st.title("🏋️‍♀️ AI 피트니스 코치")
    st.info("AI 코치가 카메라를 통해 여러분의 자세를 분석하고 실시간으로 음성 피드백을 제공합니다.")
    
    # 4. 스쿼트 유튜브 영상 링크 수정 (오류 없는 표준 링크)
    squat_video_url = "https://youtu.be/HTthpM84ILQ?si=XEjlLeZMYvRdyjsH"
    
    col1, col2 = st.columns([2, 1.5])
    with col1:
        st.subheader("🤖 실시간 스쿼트 코칭")
        webrtc_streamer(key="squat-coach", video_processor_factory=lambda: PoseAnalyzerTransformer(mode='squat'), 
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}), 
            media_stream_constraints={"video": True, "audio": False})
    with col2:
        st.subheader("📌 스쿼트 가이드 영상")
        st.video(squat_video_url)

# 1. 재활용 가이드 기능 복구
def render_recycling_guide(api_key):
    st.title("♻️ AI 재활용 가이드")
    st.info("쓰레기 사진을 업로드하면 AI가 올바른 분리배출 방법을 알려드립니다.")
    uploaded_file = st.file_uploader("쓰레기 사진을 업로드하세요.", type=["jpg", "jpeg", "png"])
    
    if uploaded_file and api_key:
        st.image(uploaded_file, caption="분석할 이미지")
        if st.button("분리배출 방법 분석하기", use_container_width=True):
            with st.spinner("AI가 이미지를 분석 중입니다..."):
                image_bytes = uploaded_file.getvalue()
                b64_image = base64.b64encode(image_bytes).decode('utf-8')
                prompt = "이 쓰레기 이미지를 보고 재질을 분석한 다음, 어떻게 분리배출해야 하는지 단계별로 간단하고 명확하게 설명해줘."
                response = get_ai_response(api_key, [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]}])
                st.markdown(response)
    elif not api_key:
        st.warning("사이드바에 OpenAI API 키를 먼저 입력해주세요.")

# --- 메인 앱 라우터 ---
def main():
    st.set_page_config(page_title="AI 라이프스타일 코치", page_icon="🤖", layout="wide")
    
    if 'initialized' not in st.session_state:
        saved_data = load_data()
        if saved_data:
            for key, value in saved_data.items(): st.session_state[key] = value
        
        st.session_state.skincare_items = {'toner': '토너/스킨', 'lotion': '로션', 'essence': '에센스/세럼', 'cream': '크림', 'sunscreen': '선크림'}
        init_keys = {
            'food_inventory': [], 'cosmetic_inventory': [], 'cart': [], 'medications': [], 'weight_log': [], 
            'water_intake': 0, 'triggered_meds': [], 'last_fitness_feedback': "", 'user_preferences': [],
            'profile': {'height': 170.0, 'weight': 65.0, 'goal_weight': 60.0, 'gender': '여성', 'age': 30},
            'skincare_routine': {'morning': {item: False for item in st.session_state.skincare_items}, 'evening': {item: False for item in st.session_state.skincare_items}},
            'skincare_log': {}
        }
        for key, value in init_keys.items():
            if key not in st.session_state: st.session_state[key] = value
        st.session_state.initialized = True

    today_str = str(datetime.now().date())
    if 'today' not in st.session_state or st.session_state.today != today_str:
        st.session_state.today = today_str
        st.session_state.water_intake = 0
        st.session_state.triggered_meds = []
        for time_of_day in ['morning', 'evening']:
            for item in st.session_state.skincare_items:
                st.session_state.skincare_routine[time_of_day][item] = False
        st.session_state.skincare_log.setdefault(today_str, {})['morning_audio'] = False
        st.session_state.skincare_log.setdefault(today_str, {})['evening_audio'] = False
        
    with st.sidebar:
        st.title("🤖 AI 라이프스타일 코치")
        api_key = st.text_input("OpenAI API 키", type="password", help="AI 기능을 사용하려면 API 키를 입력해주세요.")
        st.divider()
        # 재활용 가이드 메뉴 추가
        menu_options = ["오늘의 종합 브리핑", "내 프로필 & 목표", "쇼핑 & 인벤토리", "AI 라이프스타일 추천", "AI 피트니스 코치", "♻️ 재활용 가이드"]
        st.session_state.menu = st.radio("메뉴", menu_options, key="menu_radio")
        st.divider()
        if st.button("내 모든 데이터 저장하기"):
            save_data(st.session_state)
            st.success("데이터가 성공적으로 저장되었습니다!")

    now = datetime.now()
    days_map = ['월', '화', '수', '목', '금', '토', '일']
    current_day_ko = days_map[now.weekday()]
    for med in st.session_state.medications:
        med_time = med.get('time')
        if isinstance(med_time, time):
            med_id = f"{med['name']}_{med_time.strftime('%H:%M')}"
            is_today = current_day_ko in med.get('days', [])
            is_time = now.hour == med_time.hour and now.minute == med_time.minute
            if is_today and is_time and med_id not in st.session_state.triggered_meds:
                med_alert = f"약 복용 시간입니다! '{med['name']}'을(를) 복용하세요."
                st.toast(f"💊 {med_alert}")
                audio_bytes = text_to_speech(med_alert)
                if audio_bytes: autoplay_audio(audio_bytes, key=f"med_{med_id}_{now.strftime('%Y%m%d')}")
                st.session_state.triggered_meds.append(med_id)

    menu = st.session_state.menu
    if menu == "오늘의 종합 브리핑": render_dashboard()
    elif menu == "내 프로필 & 목표": render_profile(api_key)
    elif menu == "쇼핑 & 인벤토리": render_inventory(api_key)
    elif menu == "AI 라이프스타일 추천": render_recommendations(api_key)
    elif menu == "AI 피트니스 코치": render_fitness_coach()
    elif menu == "♻️ 재활용 가이드": render_recycling_guide(api_key)

if __name__ == "__main__":
    main()


