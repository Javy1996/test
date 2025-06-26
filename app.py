import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
import openai

# Configuración de la interfaz
st.set_page_config(page_title="Chatbot Minería Rápido", layout="centered")
st.title("⛏️ Chatbot de Minería - Optimizado")

# Historial de preguntas
if "historial" not in st.session_state:
    st.session_state.historial = []

# Sidebar
st.sidebar.header("📜 Historial")
if st.sidebar.button("🧹 Limpiar historial"):
    st.session_state.historial.clear()
    st.experimental_rerun()

for i, pregunta in enumerate(reversed(st.session_state.historial), 1):
    if st.sidebar.button(f"{i}. {pregunta[:40]}..."):
        st.session_state["pregunta_actual"] = pregunta

# Cargar API Key desde entorno seguro
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuración del modelo optimizado
Settings.llm = OpenAI(
    api_key=openai.api_key,
    model="gpt-3.5-turbo-0125",  # Más rápido
    temperature=0.5              # Más directo
)

# Cargar o crear índice
@st.cache_resource
def cargar_indice():
    if os.path.exists("storage"):
        storage = StorageContext.from_defaults(persist_dir="storage")
        return load_index_from_storage(storage).as_query_engine(
            similarity_top_k=2,
            response_mode="no_text",
            return_source=True
        )
    else:
        docs = SimpleDirectoryReader("docs_mineria").load_data()
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir="storage")
        return index.as_query_engine(
            similarity_top_k=2,
            response_mode="no_text",
            return_source=True
        )

query_engine = cargar_indice()

# Entrada de pregunta
pregunta = st.text_input("Pregunta técnica de minería:", value=st.session_state.get("pregunta_actual", ""))

if pregunta:
    with st.spinner("Procesando..."):
        raw_response = query_engine.query(pregunta)
        contexto = str(raw_response)

        # Guardar en historial
        if pregunta not in st.session_state.historial:
            st.session_state.historial.append(pregunta)
        st.session_state["pregunta_actual"] = pregunta

        # Pregunta al modelo
        try:
            respuesta = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "Eres un experto en minería. Responde de forma técnica, clara y breve."},
                    {"role": "user", "content": f"Usa este contexto:\n{contexto}\n\nPregunta:\n{pregunta}"}
                ]
            ).choices[0].message.content

            st.markdown("### 📘 Respuesta:")
            st.write(respuesta)

        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    st.info("Escribe tu pregunta relacionada con minería para comenzar.")       
