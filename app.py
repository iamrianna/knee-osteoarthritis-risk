import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Knee Osteoarthritis Risk & Arthroplasty Analyzer")
st.write("A web prototype for clinical risk prediction and X-ray hardware/bone analysis. by Rianna Tanase")

# --- SIDEBAR OR SECTION FOR CLINICAL DATA ---
st.header("1. Patient Clinical Risk Assessment")

np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Age': np.random.randint(20, 85, size=n_samples),
    'BMI': np.random.uniform(18.5, 40.0, size=n_samples),
    'Gender': np.random.choice([0, 1], size=n_samples),
    'Previous_Injury': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    'Knee_Pain_Score': np.random.randint(0, 10, size=n_samples)
})

risk_formula = (
    (data['Age'] > 60).astype(int) * 2 +
    (data['BMI'] > 30).astype(int) * 3 +
    (data['Previous_Injury'] * 4) +
    (data['Knee_Pain_Score'] > 5).astype(int) * 3 +
    np.random.normal(0, 1, size=n_samples)
)

data['OA_Risk'] = (risk_formula > 7).astype(int)

X = data.drop('OA_Risk', axis=1)
y = data['OA_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Input controls for custom patient prediction
col_a, col_b, col_c = st.columns(3)
with col_a:
    p_age = st.slider("Age", 40, 85, 65)
    p_bmi = st.slider("BMI", 18.5, 40.0, 27.0)
with col_b:
    p_gender = st.selectbox("Gender", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    p_injury = st.selectbox("Previous Injury", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
with col_c:
    p_pain = st.slider("Knee Pain Score (0-10)", 0, 10, 6)

new_patient = [[p_age, p_bmi, p_gender, p_injury, p_pain]]
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

if prediction[0] == 1:
    st.error(f"⚠️ RESULT: High Risk of Knee Osteoarthritis (Confidence: {probability[0][1]*100:.1f}%)")
else:
    st.success(f"✅ RESULT: Low/Normal Risk of Knee Osteoarthritis (Confidence: {probability[0][0]*100:.1f}%)")


# --- X-RAY PROCESSOR SECTION ---
st.markdown("---")
st.header("2. Arthroplasty X-Ray Processor")
uploaded_file = st.file_uploader("Upload Knee X-Ray Image (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # CLAHE processing
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    edges = cv2.Canny(enhanced_img, 100, 200)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img, caption="Original X-Ray", width="stretch")
        
    with col2:
        st.image(enhanced_img, caption="Contrast Enhanced", width="stretch")
        
    with col3:
        st.image(edges, caption="Hardware Edges", width="stretch")

col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img, caption="Original X-Ray", width="stretch")
        
    with col2:
        st.image(enhanced_img, caption="Contrast Enhanced", width="stretch")
        
    with col3:
        st.image(edges, caption="Hardware Edges", width="stretch")

    # --- NEW: QUANTITATIVE METrics EXTRACTION ---
    st.markdown("### 📊 Quantitative X-Ray Metrics")
    
    # 1. Estimate Tibial Component Angle using Hough Lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=40, maxLineGap=10)
    estimated_angle = 0.0
    
    if lines is not None:
        tray_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Filter for relatively horizontal lines (typical of tibial trays)
                if abs(deg) < 25:
                    tray_angles.append(deg)
        if tray_angles:
            estimated_angle = float(np.mean(tray_angles))

    # 2. Calculate Edge Density (proxy for interface complexity/wear)
    edge_pixel_count = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = (edge_pixel_count / total_pixels) * 100

    # Display Metrics in Clean Columns
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            label="Est. Tibial Tray Angle", 
            value=f"{estimated_angle:.1f}°", 
            delta="Target: 0.0° (Neutral)",
            delta_value="inverse"
        )
        
    with metric_col2:
        st.metric(
            label="Hardware Edge Density", 
            value=f"{edge_density:.2f}%",
            help="Percentage of edge pixels detected along the implant-bone interface."
        )
        
    with metric_col3:
        # Risk heuristic based on edge complexity
        interface_status = "Normal Fixation" if edge_density < 3.5 else "Potential Radiolucency / Wear"
        st.metric(
            label="Interface Assessment", 
            value=interface_status
        )
