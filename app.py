import streamlit as st

# Setup and hide sidebar
st.set_page_config(
    page_title="Smart Crowd & Bus Monitoring",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="collapsedControl"] { display: none; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    header, footer {visibility: hidden;}

    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: 0.3s ease-in-out;
        text-align: center;
        cursor: pointer;
        border: 2px solid transparent;
        text-decoration: none;
        color: inherit;
        display: block;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        border-color: #4F8BF9;
    }
    .icon {
        font-size: 48px;
        margin-bottom: 1rem;
        color: #4F8BF9;
    }
    .title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333333;
        margin-bottom: 0.5rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center; color: #4F8BF9;'>🧭 Smart Crowd & Bus Monitoring</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

# 🕋 Card 1 - with redirect (form based click)
with col1:
    st.markdown(
        """
        <form action="/crowd_in_mecca" target="_self">
            <button type="submit" style="all: unset; width: 100%;">
                <div class="card">
                    <div class="icon">🕋</div>
                    <div class="title">Crowd in Mecca</div>
                    <p>Real-time crowd analysis at holy sites </p>
                </div>
            </button>
        </form>
    """,
        unsafe_allow_html=True,
    )

# 🚌 Card 2 - placeholder (no link yet)
with col2:
    st.markdown(
        """  <form action="/bus_detection" target="_self">
            <button type="submit" style="all: unset; width: 100%;">
        <div class="card">
            <div class="icon">🚌</div>
            <div class="title">Vehicles Detection</div>
            <p>Automatic bus recognition and tracking </p> 
        </div> </button>
        </form>
    """,
        unsafe_allow_html=True,
    )

# 👥 Card 3 - placeholder (no link yet)
with col3:
    st.markdown(
        """
        <div class="card">
            <div class="icon">👥</div>
            <div class="title">People in Bus</div>
            <p>Detection of people inside bus.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

#  Card 4 - Luggage Detection

with col4:
    st.markdown(
        """
        <form action="/luggage_detection" target="_self">
            <button type="submit" style="all: unset; width: 100%;">
                <div class="card">
                    <div class="icon">🛅</div>
                    <div class="title">Luggage Detection</div>
                    <p>Detect abondoned luggage and bags</p>
                </div>
            </button>
        </form>
    """,
        unsafe_allow_html=True,
    )
