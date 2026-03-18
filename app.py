import streamlit as st
import pandas as pd
import requests
from huggingface_hub import hf_hub_download, login
login()

st.set_page_config(page_title="Corrosion Labeling Tool", layout="wide")

HF_REPO_ID = "NisrineO/CorrosionDetection"
TOTAL_BATCHES = 80

@st.cache_data
def load_batch_data(batch_num):
    df = pd.read_csv(f"expert_batches/batch_{batch_num:03d}.csv")
    return df['sample_id'].tolist()

if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = None
if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = 0
if 'batch_annotations' not in st.session_state:
    st.session_state.batch_annotations = {}

def get_next_batch():
    try:
        response = requests.get("https://script.google.com/macros/s/AKfycbz3lUi7brxYAEugXZ6A7WnqlgObUzoT8n3oeM84da6yE6L3-FrH-8EaFfDGAA9_tFGoiw/exec")
        return int(response.text)
    except Exception as e:
        st.error(f"Could not connect to database: {e}")
        return 0


def go_previous(image_name):
    st.session_state.batch_annotations[image_name] = st.session_state[f"radio_{image_name}"]

    if st.session_state.current_image_index > 0:
        st.session_state.current_image_index -= 1

def go_next(image_name):
    st.session_state.batch_annotations[image_name] = st.session_state[f"radio_{image_name}"]
    batch_images = load_batch_data(st.session_state.current_batch)
    if st.session_state.current_image_index < len(batch_images) - 1:
        st.session_state.current_image_index += 1

def finish_and_save(image_name):
    st.session_state.batch_annotations[image_name] = st.session_state[f"radio_{image_name}"]

    batch_images = load_batch_data(st.session_state.current_batch)
    if len(st.session_state.batch_annotations) < len(batch_images):
        st.error("⚠️ Error: Not all images in this batch have a score. Please use the Previous button to ensure no images were skipped.")
        return
    
    rows_to_add = []
    for filename, final_score in st.session_state.batch_annotations.items():
        rows_to_add.append({
            "Expert_Username": st.session_state.user,
            "Batch": st.session_state.current_batch,
            "Filename": filename,
            "Corrosion_Score": final_score
        })

    with st.spinner("Saving annotations to Google Sheets..."):
        GOOGLE_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbz3lUi7brxYAEugXZ6A7WnqlgObUzoT8n3oeM84da6yE6L3-FrH-8EaFfDGAA9_tFGoiw/exec"
        
        try:
            response = requests.post(GOOGLE_WEB_APP_URL, json=rows_to_add)
            if response.text == "Success":
                st.success("Annotations saved successfully!")
                st.session_state.current_batch = get_next_batch()
                st.session_state.current_image_index = 0
                st.session_state.batch_annotations = {}
            else:
                st.error("Failed to save. Please try again.")
        except Exception as e:
            st.error(f"Connection error: {e}")


def exit_batch_early():
    st.session_state.page = 'login'
    st.rerun()


if st.session_state.page == 'login':
    st.title("Corrosion Labeling Tool")
    st.write("Enter your ID to receive an assigned batch. / Inserisci il tuo ID per ricevere un batch assegnato.")
    
    expert_username = st.text_input("Enter Expert Username:")

    if st.button("Login & Get Assigned Batch"):
        if expert_username.strip() == "":
            st.warning("Please enter your ID first. / Per favore, inserisci prima il tuo ID.")
        else:
            st.session_state.user = expert_username.strip()
            with st.spinner("Assigning you a batch..."):
                st.session_state.current_batch = get_next_batch()

            st.session_state.current_image_index = 0
            st.session_state.batch_annotations = {}
            st.session_state.page = 'labeling'
            st.rerun()

elif st.session_state.page == 'labeling':
    batch_num = st.session_state.current_batch
    img_idx = st.session_state.current_image_index
    batch_images = load_batch_data(batch_num)
    current_image_name = batch_images[img_idx-1]

    col_quit, col_prog = st.columns([1, 3])
    with col_quit:
        if st.button("🚪 Save & Exit for Now"):
            st.session_state.page = 'login'
            st.rerun()
    with col_prog:
        st.progress((img_idx) / len(batch_images), text=f"Batch {batch_num:03d} | Image {img_idx + 1} of {len(batch_images)}")

    st.info("""
    **EN:** In the images are shown cementitious composite specimens, which may or may not present visible signs of surface corrosion. Please rate each image on a 0–3 scale as follows:  
    **IT:** Nelle immagini sono mostrati provini in composito cementizio, che possono presentare o meno segni visibili di corrosione superficiale. Si prega di valutare ciascuna immagine su una scala 0–3 come segue:
    """)

    with st.spinner("Loading image..."):
        img_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"{current_image_name}.png",
            repo_type="dataset"
        )
        st.image(img_path, use_column_width=True)

    st.divider()

    col_legend, col_radio = st.columns([1, 2])

    with col_legend:
        st.markdown("""
        **Legend / Legenda:**
        * **0** - Zero to low severity / Severità della corrosione da nulla a bassa
        * **1** - Low to moderate corrosion severity / Severità della corrosione da bassa a moderata
        * **2** - Moderate to high corrosion severity /  Severità della corrosione da moderata ad alta
        * **3** - High corrosion severity / Severità della corrosione alta
        """)

    with col_radio:
        st.subheader("Corrosion Assessment")
        saved_score = st.session_state.batch_annotations.get(current_image_name, 0)

        st.radio(
            "Select severity score:",
            options=[0, 1, 2, 3],
            index=saved_score,
            key=f"radio_{current_image_name}",
            horizontal=True,
            label_visibility="collapsed"
        )

    st.write("")


    nav_col1, nav_col2 = st.columns(2)

    with nav_col1:
        st.button(
            "⬅️ Previous",
            on_click=go_previous,
            args=(current_image_name,),
            disabled=(img_idx == 0),
            width=150
        )

    with nav_col2:
        is_last = (img_idx == len(batch_images) - 1)
        if is_last:
            st.button(
                "Finish & Save ✅",
                type="primary",
                on_click=finish_and_save,
                args=(current_image_name,),
                width=150,
            )
        else:
            st.button(
                "Next ➡️",
                type="primary",
                on_click=go_next,
                args=(current_image_name, ),
                width=150,
                
            )
