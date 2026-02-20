import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="SkyPredict Pro", page_icon="✈️", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    encoders = joblib.load('encoding.pkl') 
    scaler = joblib.load('scaling.pkl')
    df_eda = pd.read_csv('df_EDA.csv')
    return model, encoders, scaler, df_eda

model, encoders, scaler, df_eda = load_assets()

# --- 2. CUSTOM CSS (التصحيح هنا) ---
st.markdown("""
    <style>
    /* تغيير لون الخلفية العامة */
    .stApp { background-color: #f8f9fa; }
    
    /* تنسيق الكروت الخاصة بالمقاييس */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    /* تنسيق العناوين */
    h1, h2, h3 { color: #1e3a8a; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True) # تم تصحيح الكلمة هنا

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/784/784844.png", width=80)
    st.title("SkyControl Center")
    st.markdown("---")
    
    # حساب إحصائيات سريعة
    total = len(df_eda)
    on_time = (df_eda['Flight_Status'] == 0).sum()
    st.metric("Total Flights", f"{total:,}")
    st.metric("Accuracy Rate", f"{(on_time/total)*100:.1f}%")
    st.markdown("---")
    st.info("هذا النظام يستخدم الذكاء الاصطناعي (XGBoost) للتنبؤ بدقة الرحلات.")

# --- 4. MAIN LAYOUT ---
tab_predict, tab_viz = st.tabs(["🎯 Prediction Tool", "📊 Interactive Analytics"])

with tab_predict:
    st.subheader("🛠️ Flight Parameter Configuration")
    
    # تنظيم المدخلات في حاوية واحدة
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            year = st.selectbox("Year", [2022, 2023, 2024, 2025, 2026])
            month = st.slider("Month", 1, 12, 1)
            day_m = st.number_input("Day of Month", 1, 31, 1)
            day_w = st.selectbox("Day of Week", options=range(1, 8), 
                               format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x-1])
        with c2:
            airline = st.selectbox("Airline", sorted(df_eda['Airlines'].unique()))
            origin = st.selectbox("Origin City", sorted(df_eda['OriginCityName'].unique()))
            dest = st.selectbox("Destination City", sorted(df_eda['DestCityName'].unique()))
        with c3:
            air_time = st.number_input("Air Time (mins)", min_value=1.0, value=120.0)
            distance = st.number_input("Distance (miles)", min_value=1.0, value=500.0)
            arr_delay = st.number_input("Current ArrDelay (mins)", value=0.0)

    if st.button("🚀 Execute Smart Prediction", use_container_width=True):
        input_df = pd.DataFrame([{
            'Month': month, 'DayofMonth': day_m, 'DayOfWeek': day_w,
            'Airlines': airline, 'OriginCityName': origin, 'DestCityName': dest,
            'ArrDelay': arr_delay, 'AirTime': air_time, 'Distance': distance, 'Year': year
        }])

        try:
            # 1. Encoding
            for col in ['Airlines', 'OriginCityName', 'DestCityName']:
                input_df[col] = encoders[col].transform(input_df[[col]])
            
            # 2. Scaling
            input_df[['AirTime', 'Distance']] = scaler.transform(input_df[['AirTime', 'Distance']])
            
            # 3. Align Columns
            feature_order = ['Month', 'DayofMonth', 'DayOfWeek', 'Airlines', 'OriginCityName', 'DestCityName', 'ArrDelay', 'AirTime', 'Distance', 'Year']
            input_df = input_df[feature_order]

            # 4. Prediction
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]

            st.divider()
            
            res_c1, res_c2 = st.columns([1, 1])
            with res_c1:
                if prediction == 1:
                    st.error("### Status: DELAYED 🚩")
                    st.write(f"Confidence Level: **{prob[1]*100:.1f}%**")
                else:
                    st.success("### Status: ON TIME ✅")
                    st.write(f"Confidence Level: **{prob[0]*100:.1f}%**")
            
            with res_c2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Delay Risk %", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ef4444" if prediction == 1 else "#10b981"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ecfdf5"},
                            {'range': [50, 100], 'color': "#fef2f2"}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

with tab_viz:
    st.subheader("📊 Fleet & Route Analytics")
    v1, v2 = st.columns(2)
    
    with v1:
        # Airline Distribution
        fig_air = px.pie(df_eda, names='Airlines', title="Market Share by Airline", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_air, use_container_width=True)
        
    with v2:
        # Delay vs On-Time
        fig_status = px.histogram(df_eda, x='Airlines', color='Flight_Status', barmode='group',
                                 title="Operational Performance per Airline",
                                 color_discrete_map={0: '#10b981', 1: '#ef4444'})
        st.plotly_chart(fig_status, use_container_width=True)