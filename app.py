import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
import openai

# Configuración visual
st.set_page_config(page_title="Chatbot Minería ", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    h1 {
        color: #205375;
    }
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    #MainMenu, footer, header {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Chatbot de Minería (solo responde con documentos)")

# Historial de preguntas
if "historial" not in st.session_state:
    st.session_state.historial = []

st.sidebar.header("🕘 Historial de preguntas")
if st.sidebar.button("🗑️ Borrar historial"):
    st.session_state.historial.clear()
    st.experimental_rerun()

if st.session_state.historial:
    for i, pregunta_prev in enumerate(reversed(st.session_state.historial), 1):
        if st.sidebar.button(f"{i}. {pregunta_prev[:50]}..."):
            st.session_state["pregunta_actual"] = pregunta_prev
else:
    st.sidebar.info("No hay preguntas todavía.")

# Cargar clave API
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
Settings.llm = OpenAI(api_key=openai.api_key, model="gpt-3.5-turbo", temperature=0.9)

# Crear o cargar índice
@st.cache_resource
def cargar_indice():
    if os.path.exists("storage"):
        storage = StorageContext.from_defaults(persist_dir="storage")
        return load_index_from_storage(storage).as_query_engine(
            similarity_top_k=2,
            response_mode="compact",  # Utiliza el contenido directo del documento
            return_source=True
        )
    else:
        documentos = SimpleDirectoryReader("docs_mineria").load_data()
        index = VectorStoreIndex.from_documents(documentos)
        index.storage_context.persist(persist_dir="storage")
        return index.as_query_engine(
            similarity_top_k=2,
            response_mode="compact",
            return_source=True
        )

query_engine = cargar_indice()

# Entrada del usuario
pregunta = st.text_input("Escribe tu pregunta sobre minería:",
                         value=st.session_state.get("pregunta_actual", ""))

if pregunta:
    with st.spinner("Consultando documentos..."):
        try:
            # Recuperar respuesta directamente desde los documentos
            raw_response = query_engine.query(pregunta)
            respuesta = raw_response.response
            fuentes = raw_response.source_nodes

            if not respuesta.strip():
                respuesta = "No se encontró suficiente información en los documentos para responder esta pregunta."

            if pregunta not in st.session_state.historial:
                st.session_state.historial.append(pregunta)
            st.session_state["pregunta_actual"] = pregunta

            st.markdown("### 📘 Respuesta basada en documentos:")
            st.markdown(respuesta)

            if fuentes:
                st.markdown("---")
                st.subheader("📄 Fragmentos relevantes del contexto:")
                for i, fuente in enumerate(fuentes, 1):
                    contenido = fuente.node.get_content().strip()
                    resumen = contenido[:300].replace("\n", " ") + "..."
                    st.markdown(f"**{i}.**\n\n> {resumen}")

        except Exception as e:
            st.error(f"Error al consultar los documentos: {str(e)}")
else:
    st.info("Ingresa una pregunta para consultar el contenido de los documentos.")
    
