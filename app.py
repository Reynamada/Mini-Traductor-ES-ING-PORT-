"""
Desafio Tradutor de Voz — Streamlit App
Grava voz → Transcreve com Whisper → Traduz → Fala o resultado
"""

import io
import tempfile
import os

import streamlit as st
import whisper
from gtts import gTTS
import torch
import requests
import json
import threading

# Lock global para garantir segurança de threads na transcrição com Whisper
whisper_lock = threading.Lock()

# ─────────────────────────────────────────────
# Configuração da Página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tradutor de Voz IA",
    page_icon="🎙️",
    layout="centered",
)

# ─────────────────────────────────────────────
# CSS Customizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Fundo escuro com gradiente */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 50%, #24243e 100%);
    min-height: 100vh;
}

/* Título principal */
.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Cards de resultado */
.result-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.result-label {
    color: #a78bfa;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.result-text {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 400;
    line-height: 1.6;
}

.lang-badge {
    display: inline-block;
    background: linear-gradient(90deg, #a78bfa33, #60a5fa33);
    border: 1px solid #a78bfa55;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.85rem;
    color: #a78bfa;
    font-weight: 500;
}

/* Esconder elementos padrão do Streamlit */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 780px;}

/* Estilizar o selectbox */
.stSelectbox label {color: #94a3b8 !important; font-size: 0.9rem !important;}

/* Estilizar botões */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
}

/* Separador */
.separator {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.3), transparent);
    margin: 1.5rem 0;
}

/* Steps */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #60a5fa;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Cache dos modelos (carrega 1 vez)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper():
    # Usando o modelo 'base' (74M params) para maior precisão de idioma e transcrição
    return whisper.load_model("base")

# ─────────────────────────────────────────────
# Configuração OpenRouter e Idiomas
# ─────────────────────────────────────────────
IDIOMAS = {
    "🇬🇧 Inglês": "en",
    "🇪🇸 Español": "es",
    "🇧🇷 Português (Brasil)": "pt",
}

LANG_NAMES = {"es": "Español", "en": "English", "pt": "Português"}

# Modelos gratuitos do OpenRouter para redundância
OPENROUTER_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen-2-7b-instruct:free",
    "openrouter/free"
]

# ─────────────────────────────────────────────
# Função de Tradução (OpenRouter com Fallback)
# ─────────────────────────────────────────────
def traduzir(texto: str, origem: str, destino: str) -> str:
    """Traduz texto usando a API do OpenRouter com fallback automático entre modelos gratuitos."""
    if origem == destino:
        return texto

    # Verificar API key
    if "openrouter" not in st.secrets or "api_key" not in st.secrets["openrouter"]:
        raise Exception(
            "A API Key do OpenRouter não está configurada! "
            "Configure-a em .streamlit/secrets.toml localmente ou no painel de Secrets del Streamlit Cloud."
        )
    
    api_key = st.secrets["openrouter"]["api_key"]
    if not api_key or api_key == "sk-or-v1-SeuTokenAqui" or api_key.strip() == "":
        raise Exception(
            "A API Key do OpenRouter fornecida é inválida ou vazia! "
            "Insira uma chave real em .streamlit/secrets.toml ou no painel de Secrets del Streamlit Cloud."
        )

    lang_names_full = {
        "es": "Español",
        "en": "English",
        "pt": "Português do Brasil (Brazilian Portuguese)"
    }

    origin_lang = lang_names_full.get(origem, origem)
    target_lang = lang_names_full.get(destino, destino)

    errors = []
    for model_name in OPENROUTER_MODELS:
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Reynamada/Mini-Traductor-ES-ING-PORT-",
                "X-Title": "Mini Tradutor de Voz IA"
            }
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator. Translate the text from the source language "
                            "to the target language. Preserve formatting, meaning, and tone. "
                            "If the target language is Portuguese, you MUST translate to natural Brazilian Portuguese (Português do Brasil). "
                            "Do NOT add any notes, introductions, explanations, or markdown code blocks around the text. "
                            "Output ONLY the raw translation."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Source Language: {origin_lang}\nTarget Language: {target_lang}\nText:\n{texto}"
                    }
                ],
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=12
            )
            
            if response.status_code == 200:
                result_json = response.json()
                if "choices" in result_json and len(result_json["choices"]) > 0:
                    translation = result_json["choices"][0]["message"]["content"].strip()
                    # Remover aspas extras que os modelos às vezes colocam
                    if translation.startswith('"') and translation.endswith('"'):
                        translation = translation[1:-1].strip()
                    if translation.startswith("'") and translation.endswith("'"):
                        translation = translation[1:-1].strip()
                    return translation
                else:
                    errors.append(f"{model_name}: Resposta vazia da API")
            else:
                errors.append(f"{model_name}: HTTP {response.status_code} - {response.text}")
        except Exception as e:
            errors.append(f"{model_name}: {str(e)}")

    raise Exception(
        f"Todos os modelos gratuitos de OpenRouter falharam ao tentar traduzir. Erros detalhados:\n" + 
        "\n".join(f"- {err}" for err in errors)
    )


def gerar_audio(texto: str, lang: str) -> bytes:
    """Gera áudio MP3 com gTTS e retorna bytes."""
    # gTTS usa 'pt' para português
    lang_map = {"pt": "pt", "es": "es", "en": "en"}
    tts = gTTS(text=texto, lang=lang_map.get(lang, "en"))
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# Interface Principal
# ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🎙️ Tradutor de Voz IA</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Grave sua voz • Transcreva • Traduza • Ouça</p>', unsafe_allow_html=True)
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ── Verificação de API Key de OpenRouter ──────
has_key = False
if "openrouter" in st.secrets and "api_key" in st.secrets["openrouter"]:
    key = st.secrets["openrouter"]["api_key"]
    if key and key != "sk-or-v1-SeuTokenAqui" and key.strip() != "":
        has_key = True

if not has_key:
    st.warning(
        "⚠️ **Chave API do OpenRouter não configurada!**\n\n"
        "Para realizar as traduções, você precisa de uma API Key do OpenRouter. "
        "Siga estes passos:\n"
        "1. Obtenha uma chave gratuita em [OpenRouter Keys](https://openrouter.ai/keys).\n"
        "2. **Localmente**: Adicione a chave no arquivo `.streamlit/secrets.toml`:\n"
        "```toml\n"
        "[openrouter]\n"
        "api_key = \"sua-chave-aqui\"\n"
        "```\n"
        "3. **Streamlit Cloud**: Adicione no painel da aplicação em **App Settings** ➔ **Secrets**."
    )

# ── Passo 1: Entrada ──────────────────────────
st.markdown('<div class="step-indicator">① SEU IDIOMA E GRAVAÇÃO</div>', unsafe_allow_html=True)

# Layout em colunas para selecionar o idioma de entrada e gravar
col_lang_in, col_audio = st.columns([1, 2])

with col_lang_in:
    idioma_entrada_label = st.selectbox(
        label="Você vai falar em:",
        options=["Auto-detectar", "🇪🇸 Español", "🇧🇷 Português (Brasil)", "🇬🇧 Inglês"],
        index=0,
        key="idioma_entrada",
    )
    
IDIOMAS_ENTRADA_MAP = {
    "Auto-detectar": None,
    "🇪🇸 Español": "es",
    "🇧🇷 Português (Brasil)": "pt",
    "🇬🇧 Inglês": "en"
}
idioma_entrada_code = IDIOMAS_ENTRADA_MAP[idioma_entrada_label]

with col_audio:
    audio_input = st.audio_input(
        label="Grave sua voz",
        key="audio_recorder",
        label_visibility="collapsed"
    )

# ── Passo 2: Idioma Destino ────────────────────
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<div class="step-indicator">② IDIOMA DE DESTINO (TRADUÇÃO)</div>', unsafe_allow_html=True)

idioma_label = st.selectbox(
    label="Traduzir para:",
    options=list(IDIOMAS.keys()),
    index=0,
    key="idioma_destino",
    label_visibility="collapsed",
)
idioma_destino_code = IDIOMAS[idioma_label]

# ── Passo 3: Processar ────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
btn_processar = st.button("🚀 Transcrever e Traduzir", key="btn_processar", use_container_width=True)

# ── Processamento ─────────────────────────────
if btn_processar:
    if audio_input is None:
        st.warning("⚠️ Por favor, grave um áudio antes de continuar.")
    else:
        # Salva o áudio num arquivo temporário para o Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_input.getvalue())
            tmp_path = tmp.name

        try:
            # Transcrição
            with st.spinner("🤖 Carregando Whisper e transcrevendo..."):
                whisper_model = load_whisper()
                
                # Configura argumentos de transcrição (força idioma se selecionado)
                transcribe_args = {}
                if idioma_entrada_code is not None:
                    transcribe_args["language"] = idioma_entrada_code
                
                # Usar um lock para evitar concorrência no Whisper
                with whisper_lock:
                    result = whisper_model.transcribe(tmp_path, **transcribe_args)
                texto_original = result["text"].strip()
                idioma_detectado = result.get("language", idioma_entrada_code)

            # Exibe transcrição
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">📝 Transcrição Original</div>
                    <div class="result-text">{texto_original}</div>
                    <br/>
                    <span class="lang-badge">🌐 Idioma detectado: {LANG_NAMES.get(idioma_detectado, idioma_detectado)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tradução
            with st.spinner(f"🔄 Traduzindo para {idioma_label}..."):
                texto_traduzido = traduzir(texto_original, idioma_detectado, idioma_destino_code)

            # Exibe tradução
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">🌍 Tradução → {idioma_label}</div>
                    <div class="result-text">{texto_traduzido}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Áudio da tradução
            with st.spinner("🔊 Gerando áudio da tradução..."):
                audio_bytes = gerar_audio(texto_traduzido, idioma_destino_code)

            st.markdown('<div class="step-indicator">③ OUÇA A TRADUÇÃO</div>', unsafe_allow_html=True)
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        except Exception as e:
            st.error(f"❌ Erro durante o processamento: {e}")
        finally:
            os.unlink(tmp_path)

# ── Rodapé ────────────────────────────────────
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.8rem;">Powered by Whisper · Helsinki-NLP · gTTS · Streamlit</p>',
    unsafe_allow_html=True,
)
