import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_openai import ChatOpenAI


# ✅ Web Search 툴 사용 LLM 생성
def create_web_search_llm(model_name="gpt-4o"):
    return ChatOpenAI(
        model_name=model_name,
        tools=[{"type": "web_search_preview"}],
        tool_choice="auto",
    )


# ✅ 프롬프트 기반 체인 생성
def create_chain(prompt, model):
    return prompt | ChatOpenAI(model_name=model) | StrOutputParser()


# ✅ Streamlit UI 설정
st.set_page_config(page_title="AIVE R&D Bot 💬", page_icon="💬")
st.title("AVE R&D Bot 💬")

# ✅ 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ✅ 대화 이력 출력
def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


# ✅ 대화 이력 추가
def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# ✅ 사이드바 구성
with st.sidebar:
    clear_btn = st.button("대화내용 초기화")
    use_web_search = st.checkbox("🔎 웹 검색 모드 활성화")
    tab1, tab2 = st.tabs(["프롬프트", "프리셋"])

    default_prompt = (
        "당신은 친절한 AI 어시스턴트 입니다. 사용자의 질문에 간결하게 답변해 주세요."
    )
    user_text_prompt = tab1.text_area("프롬프트", value=default_prompt)
    user_text_apply_btn = tab1.button("프롬프트 적용", key="apply1")
    if user_text_apply_btn:
        tab1.markdown("✅ 프롬프트가 적용되었습니다")
        prompt_template = user_text_prompt + "\n\n#Question:\n{question}\n\n#Answer:"
        prompt = PromptTemplate.from_template(prompt_template)
        st.session_state["chain"] = create_chain(prompt, "gpt-4o-mini")

    user_selected_prompt = tab2.selectbox("프리셋 선택", ["sns", "번역", "요약"])
    user_selected_apply_btn = tab2.button("프롬프트 적용", key="apply2")
    if user_selected_apply_btn:
        tab2.markdown("✅ 프롬프트가 적용되었습니다")
        prompt = load_prompt(f"prompts/{user_selected_prompt}.yaml", encoding="utf8")
        st.session_state["chain"] = create_chain(prompt, "gpt-4o-mini")

# ✅ 대화 초기화 버튼
if clear_btn:
    st.session_state["messages"].clear()

# ✅ 대화 출력
print_history()

# ✅ 초기 체인 설정
if "chain" not in st.session_state:
    prompt_template = user_text_prompt + "\n\n#Question:\n{question}\n\n#Answer:"
    prompt = PromptTemplate.from_template(prompt_template)
    st.session_state["chain"] = create_chain(prompt, "gpt-4o-mini")

# ✅ 사용자 입력 처리
if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        chat_container = st.empty()

        if use_web_search:
            # ✅ 웹 검색 포함 LLM 실행
            llm = create_web_search_llm()
            response = llm.invoke(user_input)

            try:
                # ✅ 웹 검색 결과가 list 형태일 경우 자연스럽게 텍스트만 추출
                if isinstance(response.content, list):
                    texts = [
                        item["text"]
                        for item in response.content
                        if item["type"] == "text"
                    ]
                    ai_answer = "\n\n".join(texts)
                else:
                    ai_answer = response.content
            except Exception as e:
                ai_answer = f"⚠️ 응답 파싱 중 오류 발생: {str(e)}"

            chat_container.markdown(ai_answer)
        else:
            # ✅ 기존 체인 응답 (스트리밍)
            stream_response = st.session_state["chain"].stream({"question": user_input})
            ai_answer = ""
            for chunk in stream_response:
                ai_answer += chunk
                chat_container.markdown(ai_answer)

        add_history("ai", ai_answer)
