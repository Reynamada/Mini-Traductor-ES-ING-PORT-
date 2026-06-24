# 🎙️ Tradutor de Voz IA

Aplicação web de tradução de voz em tempo real, construída com **Streamlit**, **OpenAI Whisper** e modelos **Helsinki-NLP**.

## ✨ Funcionalidades

- 🎤 **Gravação de voz** direto no browser
- 🤖 **Transcrição automática** com Whisper (detecção automática de idioma)
- 🌍 **Tradução** entre Español, English e Português (Brasil)
- 🔊 **Síntese de voz** do texto traduzido com gTTS

## 🚀 Como Executar Localmente

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar o app
streamlit run app.py
```

## 🌐 Deploy no Streamlit Cloud

1. Faça fork/clone deste repositório para o seu GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte sua conta GitHub
4. Selecione este repositório e `app.py` como ponto de entrada
5. Clique em **Deploy!**
   Deploy app: [(https://minitraductorqueen.streamlit.app/)]

## 🧠 Modelos Utilizados

| Tarefa | Modelo |
|---|---|
| Transcrição | `openai/whisper-tiny` |
| ES → EN | `Helsinki-NLP/opus-mt-es-en` |
| EN → ES | `Helsinki-NLP/opus-mt-en-es` |
| PT → EN | `Helsinki-NLP/opus-mt-ROMANCE-en` |
| EN → PT | `Helsinki-NLP/opus-mt-en-ROMANCE` |
| Síntese de voz | gTTS |

## 📁 Estrutura

```
TRADUCTOR/
├── app.py              # App principal Streamlit
├── requirements.txt    # Dependências
└── README.md           # Este arquivo
```
