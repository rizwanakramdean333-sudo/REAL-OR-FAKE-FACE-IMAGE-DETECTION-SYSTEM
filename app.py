import streamlit as st
import streamlit.components.v1 as components_v1
import numpy as np
import cv2
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore
from tensorflow.keras.models import load_model                            # type: ignore
from mtcnn.mtcnn import MTCNN
import time
import base64
from io import BytesIO
import hashlib
import sqlite3

# ==========================================
# 1. DATABASE FUNCTIONS
# ==========================================
def init_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username=? AND password=?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

def user_exists(username):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username=?', (username,))
    data = c.fetchone()
    conn.close()
    return data is not None

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

init_db()

# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="REAL OR FAKE FACE IMAGE DETECTION SYSTEM",
    layout="wide",
)

# ==========================================
# 3. SESSION STATE
# ==========================================
defaults = {
    "splash_done":       False,
    "logged_in":         False,
    "uploader_key":      0,
    "auth_mode":         "login",
    "prediction_result": None,
    "page":              "main",
    "current_image":     None,
    "gradcam_data":      None,
    "last_image_hash":   None,
    "login_submitted":   False,
    "login_error":       None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def clear_image():
    st.session_state.uploader_key      += 1
    st.session_state.prediction_result  = None
    st.session_state.current_image      = None
    st.session_state.gradcam_data       = None
    st.session_state.last_image_hash    = None

def logout():
    st.session_state.logged_in          = False
    st.session_state.prediction_result  = None
    st.session_state.current_image      = None
    st.session_state.gradcam_data       = None
    st.session_state.page               = "main"
    st.session_state.last_image_hash    = None
    st.rerun()

def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]

# ==========================================
# 4. PRE-LOAD LOGO  (shared by splash + auth so base64 is encoded once)
# ==========================================
_logo_b64 = None
try:
    _logo_img = Image.open("images.jpg")
    _logo_buf = BytesIO()
    _logo_img.save(_logo_buf, format="JPEG")
    _logo_b64 = base64.b64encode(_logo_buf.getvalue()).decode()
except Exception:
    pass

def _logo_tag(size=72):
    """Circular logo — identical style on splash + login page."""
    if _logo_b64:
        return (
            f'<img src="data:image/jpeg;base64,{_logo_b64}" '
            f'style="width:{size}px;height:{size}px;object-fit:cover;border-radius:50%;'
            f'border:3px solid #4B0082;box-shadow:0 4px 18px rgba(75,0,130,0.28);'
            f'display:block;margin:0 auto;">'
        )
    return f'<div style="font-size:{int(size*0.65)}px;text-align:center;line-height:1;">🧠</div>'

# ==========================================
# 5. SPLASH SCREEN  — colors & logo identical to login page
# ==========================================
def show_splash():
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(f"""
        <style>
            /* Strip Streamlit chrome so splash truly fills the viewport */
            #MainMenu, footer, header {{ visibility:hidden !important; }}
            .block-container {{ padding:0 !important; max-width:100vw !important; }}

            @keyframes splashPulse {{
                0%,100% {{ transform:scale(1);    opacity:.88; }}
                50%      {{ transform:scale(1.04); opacity:1;   }}
            }}
            @keyframes splashFadeIn {{
                from {{ opacity:0; transform:translateY(14px); }}
                to   {{ opacity:1; transform:translateY(0);    }}
            }}

            /* ── Same white background as login page ── */
            .splash-wrap {{
                display:flex; flex-direction:column;
                align-items:center; justify-content:center;
                min-height:100vh;
                background:#ffffff;
                padding:28px 20px; text-align:center;
                animation:splashFadeIn .45s ease both;
            }}

            /* ── Same purple gradient top bar as the login form ::before ── */
            .splash-topbar {{
                position:fixed; top:0; left:0; right:0;
                height:4px;
                background:linear-gradient(90deg,#4B0082,#7c3aed,#a855f7);
                z-index:9999;
            }}

            .splash-logo-pulse {{ animation:splashPulse 1.5s ease-in-out infinite; margin-bottom:26px; }}

            /* ── Title: same #4B0082 colour, weight & case as login h1 ── */
            .splash-title {{
                color:#4B0082; font-size:20px; font-weight:800;
                letter-spacing:2px; text-transform:uppercase;
                font-family:sans-serif; margin:0 0 4px;
                
            }}

            /* ── Sub: same #999 colour & letter-spacing as login sub ── */
            .splash-sub {{
                color:#999; font-size:11px; letter-spacing:1.5px;
                text-transform:uppercase; margin:0 0 14px; font-weight:400;
            }}

            /* ── Divider: same 44 px purple bar as login page ── */
            .splash-divider {{
                width:44px; height:3px; margin:0 auto 18px; border-radius:2px;
                background:linear-gradient(90deg,#4B0082,#7c3aed);
            }}

            /* ── "Initializing" pill ── */
            .splash-pill {{
                display:inline-block;
                background:rgba(75,0,130,.07);
                border:1.5px solid rgba(75,0,130,.18);
                border-radius:20px; padding:6px 20px;
                color:#4B0082; font-size:12px; font-weight:600; letter-spacing:1px;
            }}

            @media (max-width:768px) {{
                .splash-title {{ font-size:15px !important; letter-spacing:1.5px !important; }}
                .splash-sub   {{ font-size:10px !important; }}
            }}
        </style>

        <div class="splash-topbar"></div>
        <div class="splash-wrap">
            <div class="splash-logo-pulse">{_logo_tag(108)}</div>
            <h1 class="splash-title">Real or Fake Face Detection System</h1>
            <p class="splash-sub">AI &nbsp;·&nbsp; Computer Vision &nbsp;·&nbsp; XAI</p>
            <div class="splash-divider"></div>
            <div class="splash-pill">⏳ &nbsp;Initializing System…</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1)
    placeholder.empty()

if not st.session_state.splash_done:
    show_splash()
    st.session_state.splash_done = True
    st.rerun()

# ==========================================
# 6. GLOBAL CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body { overflow-x:hidden; }
#MainMenu   { visibility:hidden; }
footer      { visibility:hidden; }

/* ── Tight padding — keeps dashboard screen-fit without overflow hacks ── */
.block-container {
    padding-top:0.6rem !important;
    padding-bottom:0.5rem !important;
    padding-left:1.2rem !important;
    padding-right:1.2rem !important;
    max-width:100% !important;
}

/* ── Info card ── */
.info-card {
    background:#ffffff; padding:16px 18px; border-radius:12px;
    border-left:5px solid #4B0082; box-shadow:0 4px 12px rgba(0,0,0,.05);
    margin-bottom:12px; font-size:13px; line-height:1.55;
}

/* ── Uploaded / base images — compact so dashboard fits viewport ── */
img {
    max-height:215px !important;
    object-fit:contain !important;
    width:100%;
}
/* ── Camera widget — full height so viewfinder + capture button are usable ── */
.stCameraInput > div {
    object-fit:contain !important;
    width:80%;
    max-height:none !important;
}

/* ══ BUTTON COLOURS — unchanged ══════════════════════════════ */
div.stButton > button[kind="primary"] {
    background:linear-gradient(90deg,#28a745,#20c155) !important;
    color:white !important; border:none !important;
    border-radius:8px !important; height:2.5em !important; font-weight:700 !important;
    box-shadow:0 3px 10px rgba(40,167,69,.3) !important; transition:all .2s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    background:linear-gradient(90deg,#218838,#1aad47) !important;
    box-shadow:0 5px 14px rgba(40,167,69,.45) !important; transform:translateY(-1px) !important;
}
div.stButton > button:not([kind="primary"]) {
    background:linear-gradient(90deg,#e53935,#ff5252) !important;
    color:white !important; border:none !important;
    border-radius:8px !important; height:2.5em !important; font-weight:700 !important;
    box-shadow:0 3px 10px rgba(229,57,53,.3) !important; transition:all .2s ease !important;
}
div.stButton > button:not([kind="primary"]):hover {
    background:linear-gradient(90deg,#c62828,#e53935) !important;
    box-shadow:0 5px 14px rgba(229,57,53,.45) !important; transform:translateY(-1px) !important;
}

/* ── Auth form submit buttons ── */
div[data-testid="stForm"] div[data-testid="stFormSubmitButton"]:nth-of-type(1) > button {
    background:linear-gradient(90deg,#4B0082,#7c3aed) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    height:2.5em !important; font-weight:700 !important; font-size:14px !important;
    box-shadow:0 3px 12px rgba(75,0,130,.35) !important;
    width:100% !important; transition:all .2s ease !important;
}
div[data-testid="stForm"] div[data-testid="stFormSubmitButton"]:nth-of-type(1) > button:hover {
    background:linear-gradient(90deg,#3a0066,#6d28d9) !important;
    box-shadow:0 5px 18px rgba(75,0,130,.5) !important; transform:translateY(-1px) !important;
}
div[data-testid="stForm"] div[data-testid="stFormSubmitButton"]:nth-of-type(2) > button {
    background:white !important; color:#4B0082 !important;
    border:2px solid #4B0082 !important; border-radius:8px !important;
    height:2.5em !important; font-weight:700 !important; font-size:14px !important;
    width:100% !important; transition:all .2s ease !important;
}
div[data-testid="stForm"] div[data-testid="stFormSubmitButton"]:nth-of-type(2) > button:hover {
    background:#f5f0ff !important; border-color:#7c3aed !important;
    transform:translateY(-1px) !important;
}

/* ── Form card ── */
div[data-testid="stForm"] {
    background:#ffffff !important; border:2px solid rgba(75,0,130,.2) !important;
    border-top:4px solid #4B0082 !important; border-radius:17px !important;
    padding:18px 22px 16px !important; box-shadow:0 8px 30px rgba(75,0,130,.12) !important;
}
div[data-testid="stForm"] input[type="text"],
div[data-testid="stForm"] input[type="password"] {
    background:#f9f7ff !important; color:#1a1a1a !important;
    border:2px solid #c4a8f0 !important; border-radius:8px !important;
    font-size:14px !important; padding:8px 12px !important;
}
div[data-testid="stForm"] input[type="text"]:focus,
div[data-testid="stForm"] input[type="password"]:focus {
    border-color:#4B0082 !important;
    box-shadow:0 0 0 3px rgba(75,0,130,.12) !important; background:#ffffff !important;
}
div[data-testid="stForm"] input::placeholder { color:#aaa !important; }
div[data-testid="stForm"] label { color:#4B0082 !important; font-weight:600 !important; font-size:13px !important; }

/* ── Columns: always align to top ── */
[data-testid="stHorizontalBlock"] { align-items:flex-start; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 7. JS — mobile detection + responsive CSS injected into parent document
# ==========================================
st.components.v1.html("""
<script>
(function () {
    function applyClass() {
        var mob = window.innerWidth <= 768;
        try {
            var pdoc = window.parent.document;
            pdoc.body.classList.toggle('is-mobile',  mob);
            pdoc.body.classList.toggle('is-desktop', !mob);
        } catch (e) {}
    }
    applyClass();
    window.addEventListener('resize', applyClass);

    try {
        var pdoc = window.parent.document;
        if (pdoc.getElementById('rof-style')) return;
        var s = pdoc.createElement('style');
        s.id  = 'rof-style';
        s.textContent = `

/* ══════════════════════════════════════════════════════════
   DESKTOP — vertically centre the auth form
══════════════════════════════════════════════════════════ */
body.is-desktop section[data-testid="stMain"] > div > div {
    display:flex;
    flex-direction:column;
    justify-content:center;
    min-height:calc(100vh - 48px);
}

/* ══════════════════════════════════════════════════════════
   MOBILE  ≤ 768 px
══════════════════════════════════════════════════════════ */
body.is-mobile .block-container {
    padding-top:0.35rem !important;
    padding-left:0.55rem !important;
    padding-right:0.55rem !important;
    padding-bottom:0.4rem !important;
}

/* Stack all Streamlit columns */
body.is-mobile [data-testid="stHorizontalBlock"] {
    flex-wrap:wrap !important;
    flex-direction:column !important;
    gap:0 !important;
}
body.is-mobile [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
    flex:0 0 100% !important; width:100% !important;
    max-width:100% !important; min-width:100% !important;
    padding-left:0 !important; padding-right:0 !important;
}

/* Auth: hide outer spacer cols, keep middle */
body.is-mobile [data-testid="stHorizontalBlock"]
    > [data-testid="stVerticalBlock"]:first-child:not(:only-child) { display:none !important; }
body.is-mobile [data-testid="stHorizontalBlock"]
    > [data-testid="stVerticalBlock"]:last-child:not(:only-child)  { display:none !important; }
body.is-mobile [data-testid="stHorizontalBlock"]
    > [data-testid="stVerticalBlock"]:nth-child(2) {
    display:block !important; flex:0 0 100% !important;
    width:100% !important; max-width:100% !important;
}

/* Dashboard: hide info col */
body.is-mobile .hide-on-mobile { display:none !important; }

/* Mobile topbar — show/hide */
body.is-mobile  #mob-topbar-el { display:flex !important; }
body.is-desktop #mob-topbar-el { display:none !important; }

/* Dashboard h1 */
body.is-mobile section[data-testid="stMain"] h1 {
    font-size:13px !important; letter-spacing:0.5px !important;
    margin-bottom:4px !important; margin-top:2px !important;
}

/* Buttons */
body.is-mobile div.stButton > button { min-height:44px !important; font-size:14px !important; }

/* Images — uploaded preview only (not camera) */
body.is-mobile img { max-height:190px !important; }
/* Camera widget — keep full height on mobile too */
body.is-mobile .stCameraInput > div { max-height:none !important; }

/* Placeholder */
body.is-mobile [style*="border:2px dashed"] { height:100px !important; font-size:12px !important; }

/* File uploader / info box */
body.is-mobile [data-testid="stFileUploader"] { font-size:13px !important; }
body.is-mobile [data-testid="stInfo"]         { font-size:13px !important; padding:8px !important; }

/* Auth form card */
body.is-mobile [data-testid="stForm"] { padding:14px 14px 12px !important; }

/* ── Explanation page ── */
body.is-mobile .exp-hero        { padding:20px 16px 18px !important; border-radius:14px !important; margin-bottom:14px !important; }
body.is-mobile .exp-hero h1     { font-size:17px !important; letter-spacing:1px !important; }
body.is-mobile .verdict-badge   { font-size:13px !important; padding:7px 16px !important; margin-top:10px !important; }
body.is-mobile .explain-card    { padding:16px 14px !important; font-size:13px !important; }
body.is-mobile .explain-card li { font-size:12px !important; margin-bottom:6px !important; }
body.is-mobile .tech-table thead th { padding:7px 8px !important; font-size:10px !important; }
body.is-mobile .tech-table tbody td { padding:7px 8px !important; font-size:10px !important; }
body.is-mobile .img-card img    { max-height:180px !important; }
body.is-mobile .img-card-label  { font-size:10px !important; }
body.is-mobile .region-card     { padding:12px 10px !important; }
body.is-mobile h3[style*="Syne"]{ font-size:13px !important; }
body.is-mobile div[style*="justify-content:space-between"] span { font-size:10px !important; }

/* Very small phones */
@media (max-width:480px) {
    body.is-mobile section[data-testid="stMain"] h1 { font-size:11px !important; }
    body.is-mobile div.stButton > button { font-size:12px !important; min-height:40px !important; }
    body.is-mobile img { max-height:155px !important; }
    body.is-mobile .exp-hero h1     { font-size:14px !important; }
    body.is-mobile .verdict-badge   { font-size:11px !important; padding:5px 10px !important; }
}
        `;
        pdoc.head.appendChild(s);
    } catch (e) {}
})();
</script>
""", height=0)

# ==========================================
# 8. AUTHENTICATION
# ==========================================
if not st.session_state.logged_in:

    st.markdown("""
    <style>
    @keyframes shake {
        0%,100% { transform:translateX(0); }
        20%      { transform:translateX(-6px); }
        40%      { transform:translateX(6px);  }
        60%      { transform:translateX(-4px); }
        80%      { transform:translateX(4px);  }
    }
    .shake { animation:shake .45s ease; }

    /* Same purple gradient top bar on the form card as on the splash screen */
    div[data-testid="stForm"]::before {
        content:''; display:block; height:5px;
        margin:-18px -22px 16px; border-radius:14px 14px 0 0;
        background:linear-gradient(90deg,#4B0082,#7c3aed,#a855f7);
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.auth_mode == "login":
        _, col_m, _ = st.columns([1, 1.05, 1])
        with col_m:
            err = st.session_state.get("login_error")
            st.markdown(f"""
            <div style="text-align:center;padding:20px 0 12px;">
                {_logo_tag(72)}
                <h1 style="font-size:19px;font-weight:800;color:#4B0082;
                           letter-spacing:2px;margin:10px 0 2px;
                           text-transform:uppercase;font-family:sans-serif;">
                    Real or Fake Face Image Detection System
                </h1>
                <p style="color:#999;font-size:11px;letter-spacing:1.5px;
                          text-transform:uppercase;margin:0;">
                    AI &nbsp;·&nbsp; Computer Vision &nbsp;·&nbsp; XAI
                </p>
                <div style="width:44px;height:3px;margin:10px auto 0;border-radius:2px;
                            background:linear-gradient(90deg,#4B0082,#7c3aed);"></div>
            </div>
            """, unsafe_allow_html=True)

            with st.form(key="login_form"):
                st.markdown(
                    "<p style='color:#4B0082;font-weight:700;font-size:15px;"
                    "margin:0 0 12px;letter-spacing:.5px;'>🔐 &nbsp;Sign in to your account</p>",
                    unsafe_allow_html=True,
                )
                user = st.text_input("Username", placeholder="Enter your username",
                                     label_visibility="collapsed")
                st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
                pswd = st.text_input("Password", type="password",
                                     placeholder="Enter your password",
                                     label_visibility="collapsed")

                if err:
                    st.markdown(f"""
                    <div class="shake"
                         style="display:flex;align-items:center;gap:8px;
                                background:linear-gradient(90deg,#fff5f5,#fff0f0);
                                border-left:4px solid #e53935;border-radius:8px;
                                padding:8px 12px;margin:8px 0 4px;">
                        <span style="font-size:16px;">⚠️</span>
                        <span style="color:#c53030;font-size:13px;font-weight:600;">{err}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    login_btn  = st.form_submit_button("🔑 &nbsp;Login",          use_container_width=True)
                with b2:
                    switch_btn = st.form_submit_button("📝 &nbsp;Create Account", use_container_width=True)
                st.markdown("""
                <p style="text-align:center;color:#bbb;font-size:11px;margin:8px 0 0;">
                    Press <b style='color:#888;'>Enter</b> to sign in instantly
                </p>
                """, unsafe_allow_html=True)

            if login_btn:
                if login_user(user, make_hashes(pswd)):
                    st.session_state.logged_in   = True
                    st.session_state.user        = user
                    st.session_state.login_error = None
                    st.rerun()
                else:
                    st.session_state.login_error = "Incorrect username or password."
                    st.rerun()

            if switch_btn:
                st.session_state.login_error = None
                st.session_state.auth_mode   = "signup"
                st.rerun()

    else:  # Signup
        _, col_m, _ = st.columns([1, 1.05, 1])
        with col_m:
            st.markdown(f"""
            <div style="text-align:center;padding:20px 0 12px;">
                {_logo_tag(72)}
                <h1 style="font-size:19px;font-weight:800;color:#4B0082;
                           letter-spacing:2px;margin:10px 0 2px;
                           text-transform:uppercase;font-family:sans-serif;">
                    Real or Fake Face Image Detection System
                </h1>
                <p style="color:#999;font-size:11px;letter-spacing:1.5px;
                          text-transform:uppercase;margin:0;">
                    AI &nbsp;·&nbsp; Computer Vision &nbsp;·&nbsp; XAI
                </p>
                <div style="width:44px;height:3px;margin:10px auto 0;border-radius:2px;
                            background:linear-gradient(90deg,#4B0082,#7c3aed);"></div>
            </div>
            """, unsafe_allow_html=True)

            with st.form(key="signup_form"):
                st.markdown(
                    "<p style='color:#4B0082;font-weight:700;font-size:15px;"
                    "margin:0 0 12px;letter-spacing:.5px;'>✨ &nbsp;Create a new account</p>",
                    unsafe_allow_html=True,
                )
                new_user = st.text_input("New Username", key="signup_user",
                                         placeholder="Choose a username",
                                         label_visibility="collapsed")
                st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
                new_pswd = st.text_input("New Password", type="password",
                                         key="signup_pwd",
                                         placeholder="Choose a password",
                                         label_visibility="collapsed")
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    signup_btn = st.form_submit_button("✅ &nbsp;Sign Up Now", use_container_width=True)
                with b2:
                    back_btn   = st.form_submit_button("← Back",              use_container_width=True)
                st.markdown("""
                <p style="text-align:center;color:#bbb;font-size:11px;margin:8px 0 0;">
                    Already have an account? Click <b style='color:#888;'>Back</b> to login
                </p>
                """, unsafe_allow_html=True)

            if signup_btn:
                if new_user.strip() and new_pswd.strip():
                    add_userdata(new_user, make_hashes(new_pswd))
                    st.success("✅ Account created! Redirecting to login…")
                    time.sleep(1)
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.warning("⚠️ Please fill in both fields.")

            if back_btn:
                st.session_state.auth_mode = "login"
                st.rerun()

# ==========================================
# 9. GRAD-CAM HELPER FUNCTIONS
# ==========================================

def get_last_spatial_layer_name(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return model.layers[i - 1].name
    return "top_activation"


def compute_gradcam_heatmap(feature_extractor, classifier, img_preprocessed):
    last_conv_name = get_last_spatial_layer_name(feature_extractor)
    grad_model = tf.keras.Model(
        inputs=feature_extractor.input,
        outputs=[
            feature_extractor.get_layer(last_conv_name).output,
            feature_extractor.output,
        ],
    )
    img_tensor = tf.cast(np.expand_dims(img_preprocessed, axis=0), tf.float32)
    try:
        with tf.GradientTape() as tape:
            conv_outputs, _ = grad_model(img_tensor)
            tape.watch(conv_outputs)
            pooled = tf.reduce_mean(conv_outputs, axis=[1, 2])
            pred   = classifier(pooled)[0][0]
        grads = tape.gradient(pred, conv_outputs)
        if grads is None:
            raise ValueError("Gradients are None.")
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap      = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
        heatmap      = tf.nn.relu(heatmap).numpy()
    except Exception:
        conv_model = tf.keras.Model(
            inputs=feature_extractor.input,
            outputs=feature_extractor.get_layer(last_conv_name).output,
        )
        conv_out = conv_model(img_tensor)
        heatmap  = tf.reduce_mean(tf.abs(conv_out[0]), axis=-1).numpy()
    if heatmap.max() > 1e-8:
        heatmap = heatmap / heatmap.max()
    return heatmap.astype(np.float32)


def enhance_heatmap(hm_raw):
    lo, hi = np.percentile(hm_raw, 1), np.percentile(hm_raw, 99)
    hm     = np.clip(hm_raw, lo, hi)
    hm     = (hm - lo) / (hi - lo + 1e-8)
    hm     = np.power(hm, 0.45)
    hm_u8  = np.uint8(255 * hm)
    mn, mx = hm_u8.min(), hm_u8.max()
    if mx > mn:
        hm_u8 = np.uint8((hm_u8.astype(np.float32) - mn) / (mx - mn) * 255)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    hm_u8 = clahe.apply(hm_u8)
    hm_u8 = cv2.bilateralFilter(hm_u8, d=9, sigmaColor=75, sigmaSpace=75)
    return hm_u8


def gradcam_jet_colormap(hm_u8):
    jet_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)


def build_gradcam_visuals(original_pil, heatmap, size=380):
    img_rgb = np.array(original_pil.resize((size, size)))
    hm_raw  = cv2.resize(heatmap, (size, size))
    hm_u8   = enhance_heatmap(hm_raw)
    jet_map = gradcam_jet_colormap(hm_u8)

    weight = np.sqrt(hm_u8.astype(np.float32) / 255.0)[:, :, np.newaxis]
    weight = np.clip(weight, 0.0, 0.60)
    blend  = (img_rgb.astype(np.float32) * (1.0 - weight)
              + jet_map.astype(np.float32) * weight).astype(np.uint8)

    face_weight = 0.30
    pure_hm = (img_rgb.astype(np.float32) * face_weight
               + jet_map.astype(np.float32) * (1.0 - face_weight)).astype(np.uint8)

    _, thresh   = cv2.threshold(hm_u8, int(255 * 0.55), 255, cv2.THRESH_BINARY)
    kernel      = np.ones((3, 3), np.uint8)
    thresh      = cv2.dilate(thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (255, 50,  0), 3)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 0), 1)

    legend_strip = np.fliplr(np.tile(np.arange(255, -1, -1, dtype=np.uint8), (30, 1)))
    return {
        "original": Image.fromarray(img_rgb),
        "blend":    Image.fromarray(blend),
        "heatmap":  Image.fromarray(pure_hm),
        "contours": Image.fromarray(contour_img),
        "legend":   Image.fromarray(gradcam_jet_colormap(legend_strip)),
        "raw":      hm_raw,
    }


def analyze_face_regions(heatmap_full, face_results, orig_hw):
    h, w   = heatmap_full.shape
    oh, ow = orig_hw
    sx, sy = w / ow, h / oh
    regions = {}
    if not face_results:
        return regions
    face = face_results[0]
    kp   = face.get("keypoints", {})
    box  = face.get("box", [0, 0, ow, oh])
    landmark_map = {
        "left_eye":   "👁️ Left Eye",
        "right_eye":  "👁️ Right Eye",
        "nose":       "👃 Nose",
        "mouth_left": "👄 Mouth",
    }
    for key, disp in landmark_map.items():
        if key not in kp:
            continue
        kx, ky = kp[key]
        hx = min(int(kx * sx), w - 1)
        hy = min(int(ky * sy), h - 1)
        r  = max(2, int(6 * min(sx, sy)))
        patch = heatmap_full[max(0, hy - r):hy + r, max(0, hx - r):hx + r]
        if patch.size:
            regions[disp] = float(patch.mean())
    fx, fy, fw, fh = box
    fy1 = max(0, int(fy * sy));        fy2 = min(h, int((fy + fh * 0.22) * sy))
    fx1 = max(0, int(fx * sx));        fx2 = min(w, int((fx + fw) * sx))
    if fy2 > fy1 and fx2 > fx1:
        regions["🧠 Forehead"] = float(heatmap_full[fy1:fy2, fx1:fx2].mean())
    fy_c1 = max(0, int((fy + fh * 0.78) * sy))
    fy_c2 = min(h, int((fy + fh) * sy))
    if fy_c2 > fy_c1 and fx2 > fx1:
        regions["🫦 Chin/Jaw"] = float(heatmap_full[fy_c1:fy_c2, fx1:fx2].mean())
    return regions


def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ==========================================
# 10. EXPLANATION PAGE  (fully scrollable — no overflow CSS applied here)
# ==========================================
def show_explanation_page(feature_extractor, classifier, scaler, detector):
    res = st.session_state.prediction_result
    img = st.session_state.current_image

    if img is None or res is None:
        st.warning("⚠️ Nothing to explain — please run a prediction first.")
        if st.button("← Go Back"):
            st.session_state.page = "main"
            st.rerun()
        return

    label     = res["label"]
    raw_pred  = res.get("raw_pred", res.get("confidence", 0.0))
    v_color   = "#27ae60" if label == "REAL" else "#e74c3c"
    v_emoji   = "✅"       if label == "REAL" else "🤖"

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    @keyframes fadeSlideUp {
        from { opacity:0; transform:translateY(22px); }
        to   { opacity:1; transform:translateY(0);    }
    }
    @keyframes pulse-ring {
        0%   { box-shadow:0 0 0 0    rgba(75,0,130,.45); }
        70%  { box-shadow:0 0 0 14px rgba(75,0,130,0);   }
        100% { box-shadow:0 0 0 0    rgba(75,0,130,0);   }
    }
    .exp-hero {
        background:linear-gradient(135deg,#0a0015 0%,#1a003a 45%,#0d0030 100%);
        border:1px solid rgba(150,100,255,.25); border-radius:22px;
        padding:38px 44px 32px; text-align:center; margin-bottom:28px;
        animation:fadeSlideUp .55s ease both; position:relative; overflow:hidden;
    }
    .exp-hero::before {
        content:''; position:absolute; inset:0;
        background:radial-gradient(ellipse 60% 50% at 50% 0%,rgba(120,60,255,.18),transparent 70%);
        pointer-events:none;
    }
    .exp-hero h1 {
        font-family:'Syne',sans-serif; font-size:30px; font-weight:800;
        color:#ffffff; letter-spacing:3px; margin:0 0 6px;
    }
    .verdict-badge {
        display:inline-flex; align-items:center; gap:10px;
        padding:10px 28px; border-radius:40px;
        font-family:'Syne',sans-serif; font-size:22px;
        font-weight:800; letter-spacing:2px; margin-top:20px;
        animation:pulse-ring 2s infinite;
    }
    .img-card {
        background:#0f0f1a; border:2px solid rgba(255,255,255,.08);
        border-radius:16px; overflow:hidden;
        animation:fadeSlideUp .5s ease both; margin-bottom:12px;
    }
    .img-card-label {
        font-family:'Syne',sans-serif; font-size:11px;
        font-weight:700; letter-spacing:2px; text-transform:uppercase;
        padding:10px 14px 8px; border-bottom:1px solid rgba(255,255,255,.07);
    }
    .region-card {
        background:#0f0f1a; border-radius:14px; padding:16px 14px 14px;
        border:2px solid rgba(255,255,255,.08); text-align:center;
        animation:fadeSlideUp .65s ease both; margin-bottom:8px;
    }
    .region-bar-track {
        background:rgba(255,255,255,.1); border-radius:6px;
        height:7px; margin-top:10px; overflow:hidden;
    }
    .explain-card {
        border-radius:18px; padding:28px 30px;
        line-height:1.85; font-size:15px;
        animation:fadeSlideUp .7s ease both;
    }
    .explain-card ul { padding-left:18px; margin-top:14px; }
    .explain-card li { margin-bottom:10px; }
    .tech-table {
        width:100%; border-collapse:collapse;
        background:#1a1a2e; border-radius:12px; overflow:hidden;
    }
    .tech-table thead tr { background:#2d1b69; }
    .tech-table thead th {
        padding:12px 16px; font-size:13px; font-weight:700;
        color:#fff; letter-spacing:1px; text-transform:uppercase; text-align:left;
    }
    .tech-table tbody tr { border-bottom:1px solid rgba(255,255,255,.08); transition:background .15s; }
    .tech-table tbody tr:hover { background:rgba(139,92,246,.12); }
    .tech-table tbody tr:last-child { border-bottom:none; }
    .tech-table tbody td { padding:11px 16px; font-size:13.5px; color:#e8e8f0; line-height:1.5; }
    .tech-table tbody td:first-child { color:#c4b5fd; font-weight:600; width:38%; }
    .tech-table tbody td code {
        background:rgba(139,92,246,.25); color:#d4b8ff;
        padding:2px 7px; border-radius:5px; font-size:12px;
    }
    .back-btn-exp > div > button {
        background:linear-gradient(90deg,#4B0082,#7c3aed) !important;
        color:white !important; border:none !important;
        border-radius:10px !important; font-weight:700 !important;
        font-size:15px !important; padding:0 28px !important; height:3em !important;
    }
    @media (max-width:768px) {
        .exp-hero { padding:22px 18px 20px !important; border-radius:14px !important; margin-bottom:16px !important; }
        .exp-hero h1 { font-size:18px !important; letter-spacing:1px !important; }
        .verdict-badge { font-size:14px !important; padding:8px 16px !important; margin-top:12px !important; }
        .explain-card  { padding:18px 16px !important; font-size:13px !important; }
        .explain-card li { margin-bottom:8px !important; }
        .tech-table thead th { padding:8px 10px !important; font-size:11px !important; }
        .tech-table tbody td { padding:8px 10px !important; font-size:11px !important; }
        h3[style*="Syne"] { font-size:13px !important; }
    }
    @media (max-width:480px) {
        .exp-hero h1 { font-size:15px !important; }
        .verdict-badge { font-size:12px !important; padding:6px 12px !important; }
        .explain-card  { padding:14px 12px !important; font-size:12px !important; }
        .tech-table tbody td { font-size:10px !important; padding:6px 8px !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown(f"""
    <div class="exp-hero">
        <div style="color:#a78bda;font-size:13px;letter-spacing:1.5px;text-transform:uppercase;">
            Explainable AI · Grad-CAM Analysis
        </div>
        <h1>🔬 VISUAL EXPLANATION</h1>
        <p style="color:#c4b5fd;font-size:14px;margin:4px 0 0;">
            Understanding every pixel that drove the model's decision
        </p>
        <div class="verdict-badge"
             style="background:rgba(255,255,255,.07);border:2px solid {v_color};color:{v_color};">
            <span>{v_emoji}</span><span>{label} FACE DETECTED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Compute Grad-CAM
    if st.session_state.gradcam_data is None:
        with st.spinner("🔬 Computing Grad-CAM activations…"):
            img_arr  = np.array(img)
            img_r    = cv2.resize(img_arr, (224, 224)).astype(np.float32)
            img_pre  = preprocess_input(img_r.copy())
            heatmap  = compute_gradcam_heatmap(feature_extractor, classifier, img_pre)
            visuals  = build_gradcam_visuals(img, heatmap)
            face_res = detector.detect_faces(img_arr)
            hm_full  = cv2.resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))
            regions  = analyze_face_regions(hm_full, face_res, (img_arr.shape[0], img_arr.shape[1]))
            st.session_state.gradcam_data = {"visuals": visuals, "regions": regions}

    visuals = st.session_state.gradcam_data["visuals"]
    regions = st.session_state.gradcam_data["regions"]

    # Attention maps
    st.markdown("""
    <h3 style="font-family:'Syne',sans-serif;color:#c4b5fd;letter-spacing:2px;
               text-transform:uppercase;font-size:16px;margin-bottom:6px;">
        📸 Pixel-Level Attention Maps
    </h3>
    <p style="color:#888;font-size:13px;margin-bottom:14px;">
        Classic Grad-CAM JET colormap.
        <b style="color:#e74c3c;">Red/yellow = high attention.</b>
        <b style="color:#3498db;">Blue = low attention.</b>
        Original face remains visible in all views.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    for col, key, accent, lbl, tip in [
        (c1, "original", "#8b5cf6", "① ORIGINAL",        "Unmodified image fed to EfficientNetB0."),
        (c2, "blend",    "#e74c3c", "② GRAD-CAM OVERLAY", "🔵 Blue = ignored · 🟢 Mild · 🔴 Max attention."),
        (c3, "contours", "#10b981", "③ FOCUS ZONES",      "Red-orange outlines mark top-55% attention regions."),
    ]:
        with col:
            b64 = pil_to_b64(visuals[key])
            st.markdown(f"""
            <div class="img-card" style="border-color:{accent};">
                <div class="img-card-label" style="color:{accent};">{lbl}</div>
                <img src="data:image/png;base64,{b64}"
                     style="width:100%;display:block;max-height:320px;object-fit:cover;" />
                <div style="padding:8px 12px;font-size:11px;color:#888;">{tip}</div>
            </div>
            """, unsafe_allow_html=True)

    leg_b64 = pil_to_b64(visuals["legend"])
    st.markdown(f"""
    <div style="margin:18px 0 4px;">
        <div style="font-family:'Syne',sans-serif;font-size:11px;color:#a78bda;
                    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;">
            🌡️ Grad-CAM Colour Scale (JET)
        </div>
        <img src="data:image/png;base64,{leg_b64}"
             style="width:100%;height:26px;border-radius:6px;display:block;
                    border:1px solid rgba(255,255,255,.15);" />
        <div style="display:flex;justify-content:space-between;
                    font-size:12px;font-weight:600;margin-top:6px;">
            <span style="color:#3498db;">🔵 Low</span>
            <span style="color:#1abc9c;">🟢 Mild</span>
            <span style="color:#f1c40f;">🟡 Moderate</span>
            <span style="color:#e74c3c;">🔴 High</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Thermal map + regions
    st.markdown("""
    <h3 style="font-family:'Syne',sans-serif;color:#c4b5fd;letter-spacing:2px;
               text-transform:uppercase;font-size:16px;margin-bottom:6px;">
        🌡️ Thermal Map &amp; Regional Scores
    </h3>
    <p style="color:#888;font-size:13px;margin-bottom:14px;">
        Left: JET thermal view with 30% face blended in.
        Right: each facial feature scored by model attention.
    </p>
    """, unsafe_allow_html=True)

    hm_col, reg_col = st.columns([1, 2], gap="large")
    with hm_col:
        b64h = pil_to_b64(visuals["heatmap"])
        st.markdown(f"""
        <div class="img-card" style="border-color:#e74c3c;">
            <div class="img-card-label" style="color:#e74c3c;">GRAD-CAM THERMAL MAP</div>
            <img src="data:image/png;base64,{b64h}"
                 style="width:100%;display:block;max-height:300px;object-fit:cover;" />
            <div style="padding:8px 12px;font-size:11px;color:#888;">
                Blue → Cyan → Green → Yellow → Red (JET) · 30% face visible
            </div>
        </div>
        """, unsafe_allow_html=True)

    with reg_col:
        st.markdown("""
        <p style="font-family:'Syne',sans-serif;color:#c4b5fd;font-weight:700;
                  font-size:13px;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">
            Which facial region did the model focus on?
        </p>
        """, unsafe_allow_html=True)
        if regions:
            max_val  = max(regions.values()) or 1.0
            sorted_r = sorted(regions.items(), key=lambda x: x[1], reverse=True)
            for row in [sorted_r[i:i+3] for i in range(0, len(sorted_r), 3)]:
                rcols = st.columns(len(row))
                for ci, (rname, rval) in enumerate(row):
                    pct     = min(rval / max_val * 100, 100)
                    level   = "HIGH" if pct > 65 else "MED" if pct > 35 else "LOW"
                    r_color = "#e74c3c" if pct > 65 else "#f59e0b" if pct > 35 else "#27ae60"
                    emoji   = rname.split(" ")[0]
                    name    = " ".join(rname.split(" ")[1:])
                    with rcols[ci]:
                        st.markdown(f"""
                        <div class="region-card">
                            <div style="font-size:26px;">{emoji}</div>
                            <div style="font-family:'Syne',sans-serif;font-size:11px;
                                        color:#c4b5fd;font-weight:700;margin-top:4px;">{name}</div>
                            <div style="color:{r_color};font-family:'Syne',sans-serif;
                                        font-weight:800;font-size:13px;margin-top:6px;">
                                {level} · {pct:.0f}%
                            </div>
                            <div class="region-bar-track">
                                <div style="width:{pct:.0f}%;height:100%;
                                            background:{r_color};border-radius:6px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No facial landmarks detected for regional breakdown.")

    st.markdown("---")

    # Evidence summary
    top_regions = [
        " ".join(r.split(" ")[1:])
        for r, _ in sorted(regions.items(), key=lambda x: x[1], reverse=True)[:2]
    ] if regions else []
    top_text = " and ".join(top_regions) if top_regions else "key facial areas"

    st.markdown(f"""
    <h3 style="font-family:'Syne',sans-serif;color:#c4b5fd;letter-spacing:2px;
               text-transform:uppercase;font-size:16px;margin-bottom:6px;">
        📖 Why Is This Image Classified as {label}?
    </h3>
    <p style="color:#888;font-size:13px;margin-bottom:14px;">
        Evidence gathered from the Grad-CAM activation pattern and known model behaviour.
    </p>
    """, unsafe_allow_html=True)

    if label == "REAL":
        exp_bg, exp_border = "linear-gradient(135deg,#021a0f,#02280f)", "#27ae60"
        exp_title = "✅ Authentic Human Face — Evidence Summary"
        exp_intro = f"The model concentrated on <strong>{top_text}</strong>, finding consistent signatures of a genuine photograph."
        bullets = [
            ("🌿", "Natural skin micro-texture",
             "Real skin carries irregular pore patterns, fine hairs, and subtle blemishes that GAN models fail to reproduce faithfully — the model detected these cues."),
            ("💡", "Physically plausible lighting",   "Shadows and specular highlights follow a consistent real-world light source."),
            ("📐", "Organic facial geometry",          "Slight natural asymmetry of eyes and mouth matches real human anatomy."),
            ("🔍", "Coherent feature transitions",     "Edges between skin, hair, and eyes show organic gradients without GAN artifacts."),
            ("🎞️","Camera sensor noise signatures",   "Low-level frequency noise patterns match genuine photographic sensor noise."),
        ]
    else:
        exp_bg, exp_border = "linear-gradient(135deg,#1a0202,#280202)", "#e74c3c"
        exp_title = "🤖 AI-Generated Face — Evidence Summary"
        exp_intro = f"The model flagged <strong>{top_text}</strong> as the primary areas where synthetic artifacts were detectable."
        bullets = [
            ("⚠️", "Unnatural skin texture",
             "Overly smooth or repetitively patterned skin — GANs generate texture without true stochasticity, producing tell-tale uniformity."),
            ("👁️", "Eye & iris anomalies",            "Iris reflections often appear duplicated or geometrically inconsistent."),
            ("🎨", "Color bleed at boundaries",       "Artificial gradients bleed between hair, skin, and background."),
            ("🔮", "High-frequency checkerboard artifacts",
             "Upsampling artifacts leave periodic noise detectable by EfficientNet features."),
            ("📦", "Background–face boundary inconsistency",
             "The edge between face and background shows blending artifacts."),
        ]

    bullets_html = "".join([
        f'<li style="margin-bottom:13px;"><span style="font-size:18px;">{ico}</span>'
        f'<strong style="color:#e0e0e0;"> {t}:</strong>'
        f'<span style="color:#b0b0b0;"> {d}</span></li>'
        for ico, t, d in bullets
    ])
    st.markdown(f"""
    <div class="explain-card"
         style="background:{exp_bg};border:1px solid {exp_border}44;border-left:5px solid {exp_border};">
        <h4 style="font-family:'Syne',sans-serif;color:{exp_border};
                   margin-top:0;font-size:16px;letter-spacing:1px;">{exp_title}</h4>
        <p style="color:#c0c0c0;margin-bottom:14px;">{exp_intro}</p>
        <ul style="list-style:none;padding-left:0;">{bullets_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔧 Technical Details", expanded=False):
        layer_name = get_last_spatial_layer_name(feature_extractor)
        rows_data = [
            ("Model Backbone",           "EfficientNetB0 (ImageNet pretrained)"),
            ("Classifier File",          "face_classifier.h5"),
            ("Scaler File",              "scaler.pkl"),
            ("Grad-CAM Target Layer",    f"<code>{layer_name}</code>"),
            ("Input Resolution",         "224 × 224 px"),
            ("Classification Threshold", "0.50 — pred &gt; 0.5 → REAL, else FAKE"),
            ("Raw P(REAL) Score",        f"{raw_pred:.6f}"),
            ("Verdict", f'<span style="color:{v_color};font-weight:700;font-size:14px;">{label}</span>'),
            ("XAI Method",  "Gradient-weighted Class Activation Mapping (Grad-CAM)"),
            ("Colormap",    "JET — Blue → Cyan → Green → Yellow → Red"),
            ("Contrast Pipeline",
             "Percentile clip · Gamma 0.45 · Histogram stretch · CLAHE · Bilateral filter"),
            ("Overlay blend",     "√(attention) weight, max 60% — face always visible"),
            ("Thermal map blend", "70% JET + 30% original face"),
        ]
        rows_html = "".join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in rows_data])
        st.markdown(f"""
        <table class="tech-table">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="back-btn-exp">', unsafe_allow_html=True)
    if st.button("← Back to Detection", use_container_width=False, key="back_bottom"):
        st.session_state.page = "main"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 11. MAIN DASHBOARD
# ==========================================
if st.session_state.logged_in:

    @st.cache_resource
    def load_models():
        feature_extractor = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, pooling="avg"
        )
        try:
            classifier = load_model("face_classifier.h5")
            scaler     = joblib.load("scaler.pkl")
            return feature_extractor, classifier, scaler
        except Exception:
            return None, None, None

    feature_extractor, classifier, scaler = load_models()
    detector = MTCNN()

    # Route to explanation (scrollable by Streamlit default — no CSS interference)
    if st.session_state.page == "explanation":
        show_explanation_page(feature_extractor, classifier, scaler, detector)
        st.stop()

    # ── Dashboard title ──────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;color:#4B0082;margin-top:0;margin-bottom:6px;"
        "font-size:clamp(30px,1.8vw,20px);letter-spacing:2px;'>"
        "REAL OR FAKE FACE IMAGE DETECTION SYSTEM</h1>",
        unsafe_allow_html=True,
    )

    # ── Mobile topbar ────────────────────────────────────────────────────────
    # Pure HTML; shown only when body.is-mobile via JS-injected CSS.
    # Its Logout button clicks the ONE real st.button in col_info.
    # On desktop: topbar is display:none → zero duplication.
    st.markdown(f"""
    <style>
    #mob-topbar-el {{
        display:none;
        justify-content:space-between; align-items:center;
        background:linear-gradient(90deg,#4B0082,#7c3aed);
        border-radius:10px; padding:8px 14px; margin-bottom:8px; color:white;
    }}
    #mob-topbar-el .mob-info  {{ font-size:12px; line-height:1.4; }}
    #mob-topbar-el .mob-title {{ font-weight:800; letter-spacing:1.2px; text-transform:uppercase; font-size:11px; }}
    #mob-topbar-el .mob-user  {{ opacity:.85; font-size:11px; margin-top:1px; }}
    #mob-topbar-el .mob-btn   {{
        background:rgba(255,255,255,.15);
        border:1.5px solid rgba(255,255,255,.4);
        border-radius:7px; padding:6px 14px;
        font-size:12px; font-weight:700; color:white;
        cursor:pointer; transition:background .15s; min-height:36px;
    }}
    #mob-topbar-el .mob-btn:hover  {{ background:rgba(255,80,80,.5);  border-color:rgba(255,120,120,.6); }}
    #mob-topbar-el .mob-btn:active {{ background:rgba(229,57,53,.8); }}
    </style>
    <div id="mob-topbar-el">
        <div class="mob-info">
            <div class="mob-title">Real or Fake Detection</div>
            <div class="mob-user">👤 {st.session_state.user}</div>
        </div>
        <button class="mob-btn"
                onclick="(function(){{
                    try {{
                        var btns = window.parent.document.querySelectorAll('button');
                        for(var i=0;i<btns.length;i++){{
                            if(btns[i].innerText.trim().startsWith('🚪')){{
                                btns[i].click(); break;
                            }}
                        }}
                    }} catch(e) {{}}
                }})()">
            🚪 Logout
        </button>
    </div>
    """, unsafe_allow_html=True)

    # ── Three-column layout ──────────────────────────────────────────────────
    col_info, col_input, col_result = st.columns([1, 2, 2], gap="medium")

    # ── Col 1 — Info panel (desktop only) ───────────────────────────────────
    with col_info:
        st.markdown('<div class="hide-on-mobile">', unsafe_allow_html=True)
        st.markdown(
            f"<p style='margin:0 0 8px;font-size:13px;color:#555;'>"
            f"👤 <b>User:</b> {st.session_state.user}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div class="info-card">
            <h4 style="margin-top:0;margin-bottom:8px;color:#4B0082;font-size:13px;">
                📋 Best Practices
            </h4>
            <div style="line-height:1.5;font-size:12.5px;color:#333;">
                <p style="margin:0 0 6px;">💡 <b>Lighting:</b> Use consistent, bright light to avoid shadows.</p>
                <p style="margin:0 0 6px;">🎯 <b>Pose:</b> Look directly into the camera lens.</p>
                <p style="margin:0;">✨ <b>Quality:</b> Ensure the image is sharp, contains a full face and is not blurry.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # THE ONE real Logout button — visible on desktop, hidden on mobile via CSS
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Col 2 — Upload / webcam controls ────────────────────────────────────
    image    = None
    img_hash = None

    with col_input:
        st.info("👇 **Step 1: Select Input Source**")
        input_method = st.radio(
            "Source:", ("Upload Image", "Use Webcam"),
            horizontal=True, label_visibility="collapsed",
        )

        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Drop image here...", type=["jpg", "png", "jpeg"],
                key=f"up_{st.session_state.uploader_key}",
            )
            if uploaded_file:
                raw_bytes = uploaded_file.getvalue()
                img_hash  = image_hash(raw_bytes)
                if img_hash != st.session_state.last_image_hash:
                    st.session_state.prediction_result = None
                    st.session_state.gradcam_data      = None
                    st.session_state.last_image_hash   = img_hash
                image = Image.open(BytesIO(raw_bytes)).convert("RGB")
                st.session_state.current_image = image

        st.markdown("---")
        st.write("👇 **Step 2: Actions**")
        b1, b2 = st.columns(2)
        with b1:
            predict_clicked = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
        with b2:
            st.button("🗑️ Reset", on_click=clear_image, use_container_width=True)

    # ── Col 3 — Preview + result ─────────────────────────────────────────────
    with col_result:
        if input_method == "Use Webcam":
            picture = st.camera_input("Take Snapshot", key=f"cam_{st.session_state.uploader_key}")
            if picture:
                raw_bytes = picture.getvalue()
                img_hash  = image_hash(raw_bytes)
                if img_hash != st.session_state.last_image_hash:
                    st.session_state.prediction_result = None
                    st.session_state.gradcam_data      = None
                    st.session_state.last_image_hash   = img_hash
                image = Image.open(BytesIO(raw_bytes)).convert("RGB")
                st.session_state.current_image = image

        display_image = image if image is not None else st.session_state.current_image

        if input_method == "Upload Image":
            if display_image:
                st.image(display_image, caption="Preview", use_container_width=True)
            else:
                st.markdown("""
                <div style="border:2px dashed #ccc;border-radius:10px;height:175px;
                            display:flex;align-items:center;justify-content:center;
                            color:#999;background:#fcfcfc;font-size:13px;">
                    <i>Image preview will appear here</i>
                </div>
                """, unsafe_allow_html=True)

        if predict_clicked:
            active_image = image if image is not None else st.session_state.current_image
            if active_image is None:
                st.warning("⚠️ Please provide an image first!")
            else:
                with st.spinner("Analyzing…"):
                    img_array    = np.array(active_image)
                    face_results = detector.detect_faces(img_array)
                    num_faces    = len([f for f in face_results if f["confidence"] > 0.90])

                    if num_faces == 0:
                        st.session_state.prediction_result = {
                            "label": "NO FACE", "color": "#ff9800",
                            "confidence": 0.0, "raw_pred": 0.0,
                        }
                    else:
                        img_resized  = cv2.resize(img_array, (224, 224)).astype(np.float32)
                        img_pre      = preprocess_input(img_resized)
                        features     = feature_extractor.predict(np.expand_dims(img_pre, 0), verbose=0)
                        scaled_feats = scaler.transform(features)
                        pred         = classifier.predict(scaled_feats, verbose=0)[0][0]
                        label        = "REAL" if pred > 0.5 else "FAKE"
                        color        = "#28a745" if label == "REAL" else "#dc3545"
                        confidence   = float(pred) if label == "REAL" else float(1.0 - pred)
                        st.session_state.prediction_result = {
                            "label": label, "color": color,
                            "confidence": confidence, "raw_pred": float(pred),
                        }

                    st.session_state.gradcam_data = None
                st.rerun()

        if st.session_state.prediction_result:
            res = st.session_state.prediction_result
            st.markdown(f"""
            <div style="background:{res['color']};color:white;padding:6px 8px;
                        border-radius:8px;text-align:center;margin-top:8px;
                        box-shadow:0 2px 4px rgba(0,0,0,.1);">
                <h3 style="margin:0;font-size:15px;">{res['label']} FACE DETECTED</h3>
            </div>
            """, unsafe_allow_html=True)

            if res["label"] in ("REAL", "FAKE"):
                st.markdown("<div style='margin-top:6px;'>", unsafe_allow_html=True)
                if st.button(
                    f"🔬 See Explanation — Why is it {res['label']}?",
                    use_container_width=True,
                    key="goto_explanation",
                ):
                    st.session_state.page = "explanation"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)