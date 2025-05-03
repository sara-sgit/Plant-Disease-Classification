# -*- coding: utf-8 -*-
"""
Plant Disease Classification App

Streamlit application for classifying plant diseases using deep learning models.
This app was developed as part of a master's thesis project focusing on feature
selection and fusion of deep features from CNN models (VGG16, ResNet152, DenseNet201)
for plant disease classification, specifically targeting tomato diseases from PlantVillage.

Features:
- Disease prediction using two different protocols
- Statistical results visualization
- Disease symptom information
"""

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from Prediction_module import predict_disease2, predict_disease3

# App configuration
st.set_page_config(
    page_title="Plant AI",
    page_icon=":tomato:",
    initial_sidebar_state="collapsed"
)

# CSS styling
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(90deg, rgba(255,255,255,0.6783847327993697) 11%, 
    rgba(255,255,255,0.7568161053483894) 16%, rgba(0,199,63,0.2133987384016106) 100%)}
    
[data-testid="stHeader"] {
    background-color:rgba(0,0,0,0);
    }

.block-container {
    width: 100%;}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Navigation menu
st.markdown("<h3 style='text-align:center; font-family:Verdana; font-size:25px; color:#000000;'>PlantAI</h3>", 
            unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Home", "Prediction", "Statistics", "Contact"],
    icons=["house", "images", "clipboard-data", "envelope"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"display": "flex", "width": "100%", "padding": "0!important",
                     "background":"linear-gradient(90deg, rgba(255,255,255,0.6783847327993697) 11%, "
                     "rgba(255,255,255,0.7568161053483894) 16%, rgba(0,199,63,0.2133987384016106) 100%)"},
        "icon": {"color": "white", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "center", "margin":"15px"},
        "nav-link-selected": {"background-color": "#6ebba6"},
    }
)

# Helper function for disease symptoms
def show_disease_symptoms(disease_name, symptoms, image_path):
    st.write("")
    st.write("")
    st.write("")
    col6, col7, col8 = st.columns([10, 1, 4])
    col6.markdown(f"<b><span style='text-align: justify'>{symptoms}</span></b>", unsafe_allow_html=True)
    image = Image.open(image_path)
    col8.image(image, use_column_width=True)

# Home Page
if selected == "Home":
    col1, col2, col3 = st.columns([50, 1000, 50])
    image = Image.open('assets/robots.jpg')
    resized_image = image.resize((1900, 900))
    col2.image(resized_image, use_column_width=True)
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    col1, col2, col3, col4, col5 = st.columns([100, 100, 500, 1, 100])
    col1.markdown(
        "<div style='background-color: 4CAF50;'>"
        "<h3 style='font-size: 35px; font-family: Arial;'>"
        "<span style='color: #000000;'>Why</span> "
        "<span style='color: #4F9971;'>PlantAI?</span>"
        "</h3>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    col3.markdown(
        "<div style='text-align: justify; display: inline-block; width: 520px;'>"
        "<span style='font-size:18px;'>PlantAI is an interface designed to present the results of a master's study "
        "focusing on the selection and fusion of deep features from three convolutional neural network (CNN) models "
        "(VGG16, ResNet152, and DenseNet201) for plant disease classification, specifically targeting tomato diseases "
        "from the PlantVillage database.</span></div>", 
        unsafe_allow_html=True
    )
    
    col1.markdown(
        "<div style='text-align: justify; display: inline-block; width: 697px;'>"
        "<span style='font-size:18px;'>This interface allows you to explore two classification protocols using the "
        "following feature selection methods: PCA (Principal Component Analysis), RELIEF, and RFE (Recursive Feature "
        "Elimination).</span></div>", 
        unsafe_allow_html=True
    )
    
    # Disease Symptoms Section
    st.write("")
    st.write("")
    st.write("")
    
    st.markdown(
        "<div style='background-color: 4CAF50; display: inline-block; width: 900px;'>"
        "<h2 style='font-size: 21px; font-family: Arial;'>"
        "<span style='color: #000000;'>Learn about these pathologies!</span>"
        "</h2>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    col1, col2, col3, col4, col5 = st.columns([10, 10, 15, 11, 12])
    
    if col1.button("Target spot"):
        symptoms = "Symptoms presented by the leaves of this disease are: Small wet lesions appearing on the upper surface of leaves, sometimes locally limited by a vein. Rather circular spots reaching 2 cm in diameter, surrounded by a clearly visible yellow halo."
        show_disease_symptoms("Target Spot", symptoms, 'assets/Target.png')
    
    if col2.button("Mosaic virus"):
        symptoms = "Symptoms presented by the leaves of this disease are: Lightening of the veins, marbling, mosaic in green or yellow patches with the leaf blade becoming blistered and curled. Flower drop may also be observed. It is rare for all leaves to be affected."
        show_disease_symptoms("Mosaic Virus", symptoms, 'assets/mosaic.png')
    
    if col3.button("Yellow Leaf Curl Virus"):
        symptoms = "Symptoms presented by the leaves of this disease are: Leaves gradually curling upwards from the leaf blade, giving it the appearance of a spoon. More or less intense interveinal yellowing of the leaf blade which tends to harden, and sometimes take on a purplish coloration particularly at the raised veins under the leaflets."
        show_disease_symptoms("Yellow Leaf Curl Virus", symptoms, 'assets/yelloww.png')
    
    if col4.button("Bacterial spot"):
        symptoms = "Symptoms presented by the leaves of this disease are: The lower leaves are usually attacked first. Yellow spots develop on the upper surface of leaves. Pale grayish-brown mold growth is found on the corresponding underside. Heavily infected leaves turn brown and shrivel but do not fall."
        show_disease_symptoms("Bacterial Spot", symptoms, 'assets/bacterial.png')
    
    if col4.button("Early Blight"):
        symptoms = "Symptoms presented by the leaves of this disease are: Leaf spots initially dark green, quickly turning brown to black. They are more or less rounded, sometimes angular when bounded by veins. They show subtle concentric patterns giving them a target-like appearance. A more or less bright yellow halo surrounds them. They eventually become necrotic. Severely infected leaves turn brown and fall, or dry dead leaves may cling to the stem."
        show_disease_symptoms("Early Blight", symptoms, 'assets/early.png')
    
    if col2.button("Late Blight"):
        symptoms = "Symptoms presented by the leaves of this disease are: First as small water-soaked areas that quickly enlarge to form oily-appearing brown-purple spots. On the underside of leaves, whitish-gray mycelial rings and sporulated structures may appear around the spots."
        show_disease_symptoms("Late Blight", symptoms, 'assets/late.png')
    
    if col1.button("Leaf Mold"):
        symptoms = "Symptoms presented by the leaves of this disease are: begin with light green to yellow spots on the upper leaf surface which may also be noticed on the underside of leaves. On the underside of leaves, the velvety and bronzed appearance of infected sections can be noticed as the disease progresses, which may be absent at first. The spots on the underside of leaves may turn dark olive green. Leaves may curl and wilt and the disease may cause premature leaf drop."
        show_disease_symptoms("Leaf Mold", symptoms, 'assets/leaf.png')
    
    if col3.button("Septoria leaf spot"):
        symptoms = "Symptoms presented by the leaves of this disease are: The first symptoms appear as small water-soaked circular spots 1/16 to 1/8 in diameter on the underside of older leaves. The center of these spots then becomes gray to bronzed and has a dark brown margin. The spots are typically circular and are often quite numerous."
        show_disease_symptoms("Septoria Leaf Spot", symptoms, 'assets/spectoria.png')
    
    if col5.button("Two-spotted spider mite"):
        symptoms = "Symptoms presented by the leaves of this disease are: Tiny white or yellow spots, giving leaves and needles a stippled or marbled appearance. Over time plants look bronzed or whitened and leaves may fall."
        show_disease_symptoms("Two-spotted Spider Mite", symptoms, 'assets/spider.png')

# Prediction Page
elif selected == 'Prediction':
    upload_counter_1 = 0
    upload_counter_2 = 0

    approach_1_options = ["PCA before fusion", "RELIEF before fusion", "RFE before fusion"]
    approach_2_options = ["PCA after fusion", "RELIEF after fusion", "RFE after fusion"]

    # Protocol 1 - Fusion before selection
    st.markdown(
        "<div style='box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); background-color:#f4f4f4; text-align:center; padding: 10px; border-radius: 5px;'>"
        "<h2 style='color: #4F9971; font-size: 17px; font-family: Verdana;'>"
        "Prediction with Protocol 1"
        "</h2>"
        "<p style='font-family: Verdana;font-size: 14px; '>"
        "Feature fusion before selection."
        "</p>"
        "</div>", 
        unsafe_allow_html=True
    )
    st.write("")
    
    st.markdown("<p style='font-family: Verdana; font-size: 12px;'><b>Select a feature selection method:</b></p>", 
                unsafe_allow_html=True)
    selected_approach_2 = st.selectbox(" ", approach_2_options, key="approach2")
    
    st.write("<span style='font-size: 13px; font-family: Verdana;'><b>Upload your image:</b></span>",
             unsafe_allow_html=True)
    image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], key=f"image_upload_2_{upload_counter_2}")
    
    if image_file is not None:
        predict_disease3(image_file)
        upload_counter_2 += 1 if selected_approach_2 == approach_2_options[0] else 3 if selected_approach_2 == approach_2_options[1] else 5

    st.write("")
    st.write("")

    # Protocol 2 - Selection before fusion
    st.markdown(
        "<div style='box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); background-color:#f4f4f4; text-align:center; padding: 10px; border-radius: 5px;'>"
        "<h2 style='color: #4F9971; font-size: 17px; font-family: Verdana;'>"
        "Prediction with Protocol 2"
        "</h2>"
        "<p style='font-family: Verdana;font-size: 14px; '>"
        "Feature selection before fusion."
        "</p>"
        "</div>", 
        unsafe_allow_html=True
    )
    st.write("")
    
    st.markdown("<p style='font-family: Verdana; font-size: 12px;'><b>Select a feature selection method:</b></p>", 
                unsafe_allow_html=True)
    selected_approach_1 = st.selectbox(" ", approach_1_options, key="approach1")
    
    st.write("<span style='font-size: 13px; font-family: Verdana;'><b>Upload your image:</b></span>",
             unsafe_allow_html=True)
    image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], key=f"image_upload_1_{upload_counter_1}")
    
    if image_file is not None:
        predict_disease2(image_file)
        upload_counter_1 += 1 if selected_approach_1 == approach_1_options[0] else 3 if selected_approach_1 == approach_1_options[1] else 5

# Statistics Page
elif selected == 'Statistics':
    st.markdown(
        "<div>"
        "<h3 style='font-size: 17px; font-family: Verdana;'>"
        "Results Repository:"
        "</h3>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    st.write("")
    
    with st.expander("Protocol 1 Results"):
        st.write("")
        st.write("")
        st.write("")
        st.markdown("""
            <table>
                <tr>
                    <th>Activation Function</th>
                    <th>Selection Method</th>
                    <th>Selected Features</th>
                    <th>Accuracy %</th>
                    <th>Duration</th>
                </tr>
                <tr>
                    <th rowspan="3">Sigmoid</th>
                    <td>PCA</td>
                    <td>250</td>
                    <th>97.96</th>
                    <td>Few seconds</td>
                </tr>
                <tr>
                    <td>RFE</td>
                    <td>1500</td>
                    <td>97.93</td>
                    <td>20 min</td>
                </tr>
                <tr>
                    <td>RELIEF-F</td>
                    <td>800</td>
                    <td>96.06</td>
                    <td>2 hours</td>
                </tr>
                <tr>
                    <th rowspan="3">Leakyrelu</th>
                    <td>PCA</td>
                    <td>250</td>
                    <th>98.09</th>
                    <td>Few seconds</td>
                </tr>
                <tr>
                    <td>RFE</td>
                    <td>1500</td>
                    <td>98.00</td>
                    <td>20 min</td>
                </tr>
                <tr>
                    <td>RELIEF-F</td>
                    <td>800</td>
                    <td>97.68</td>
                    <td>2 hours</td>
                </tr>
            </table>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("<h6 style='text-align:left; font-family:Verdana; font-size:14px; color:#000000;'>Confusion matrices obtained by PCA method:</h6>", 
                   unsafe_allow_html=True)
        st.write("")
        
        col11, col12, col13 = st.columns([90, 5, 90]) 
        image = Image.open('assets/SIGMO_P_1.png')
        col11.image(image, use_column_width=True)
        col11.write("<center><b>Sigmoid activation function.</b></center>", unsafe_allow_html=True)
        
        image = Image.open('assets/LEAKY_P_1.png')
        col13.image(image, use_column_width=True)
        col13.write("<center><b>Leakyrelu activation function.</b></center>", unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        st.write("")
        st.write("")

    with st.expander("Protocol 2 Results"):
        st.write("")
        st.write("")
        st.write("")
        st.markdown("""
            <table>
                <tr>
                    <th>Activation Function</th>
                    <th>Selection Method</th>
                    <th>Selected Features</th>
                    <th>Accuracy %</th>
                    <th>Duration</th>
                </tr>
                <tr>
                    <th rowspan="3">Sigmoid</th>
                    <td>PCA</td>
                    <td>300</td>
                    <th>97.81</th>
                    <td>Few seconds</td>
                </tr>
                <tr>
                    <td>RFE</td>
                    <td>900</td>
                    <td>97.53</td>
                    <td>15 min</td>
                </tr>
                <tr>
                    <td>RELIEF-F</td>
                    <td>1200</td>
                    <td>97.09</td>
                    <td>30 min</td>
                </tr>
                <tr>
                    <th rowspan="3">Leakyrelu</th>
                    <td>PCA</td>
                    <td>300</td>
                    <th>98.28</th>
                    <td>Few seconds</td>
                </tr>
                <tr>
                    <td>RFE</td>
                    <td>900</td>
                    <td>98.09</td>
                    <td>15 min</td>
                </tr>
                <tr>
                    <td>RELIEF-F</td>
                    <td>900</td>
                    <td>97.87</td>
                    <td>30 min</td>
                </tr>
            </table>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("<h6 style='text-align:left; font-family:Verdana; font-size:14px; color:#000000;'>Confusion matrices obtained by PCA method:</h6>", 
                   unsafe_allow_html=True)
        st.write("")
        
        col11, col12, col13 = st.columns([90, 5, 90]) 
        image = Image.open('assets/SIGMO_P_2.png')
        col11.image(image, use_column_width=True)
        col11.write("<center><b>Sigmoid activation function.</b></center>", unsafe_allow_html=True)
        
        image = Image.open('assets/LEAKY_P_2.png')
        col13.image(image, use_column_width=True)
        col13.write("<center><b>Leakyrelu activation function.</b></center>", unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        st.write("")
        st.write("")

    with st.expander("Results Comparison"):
        st.write("")
        st.write("")
        st.write("")
        col14, col15, col16 = st.columns([5, 80, 5])
        
        image = Image.open('assets/P1_H_r.png')
        col15.image(image, use_column_width=True)
        col15.write("<center><b>Protocol 1</b></center>", unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        image = Image.open('assets/P2_H_r.png')
        col15.image(image, use_column_width=True)
        col15.write("<center><b>Protocol 2</b></center>", unsafe_allow_html=True)

# Contact Page
elif selected == 'Contact':
    st.markdown(
        """
        <div style='border: 1px solid gray; background-color: transparent; padding: 10px;'>
            <h3 style='font-weight: bold; text-align: center;'>BENNANI Sara</h3>
            <ul>
                <li style='list-style-type: disc; font-weight: bold;'>Master's student in Networks and Telecommunications at USTHB.</li>
                <li style='list-style-type: disc; font-weight: bold;'>Email: <a href='mailto:sarahbennani61@gmail.com'>sarahbennani61@gmail.com</a></li>
                <li style='list-style-type: disc; font-weight: bold;'>LinkedIn: <a href='https://www.linkedin.com/in/sarabennani-/'>Sara Bennani</a></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("")
    st.write("")
    
    st.markdown(
        """
        <div style='border: 1px solid gray; background-color: transparent; padding: 10px;'>
            <h3 style='font-weight: bold; text-align: center;'>AKRIB Faiza</h3>
            <ul>
                <li style='list-style-type: disc; font-weight: bold;'>Master's student in Networks and Telecommunications at USTHB.</li>
                <li style='list-style-type: disc; font-weight: bold;'>Email: <a href='mailto:akribfaiza@gmail.com'>akribfaiza@gmail.com</a></li>
                <li style='list-style-type: disc; font-weight: bold;'>LinkedIn: <a href='https://www.linkedin.com/in/faiza-a-60b99120b/'>Faiza Akrib</a></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )