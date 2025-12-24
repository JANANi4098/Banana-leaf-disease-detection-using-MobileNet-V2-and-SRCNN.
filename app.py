import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from PIL import Image
from streamlit_option_menu import option_menu
import base64
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

labels=['cordana','healthy','pestalotiopsis','sigatoka']
y = np.array(labels)
label_encoder=LabelEncoder()
label_encoder.fit_transform(y)

original_title = '<b><center><p style="font-family:Times new roman; color:black; font-size: 40px;">BANANA LEAF DISEASE CLASSIFIER</p></center></b>'
st.markdown(original_title, unsafe_allow_html=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:static/background.jfif;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('static/background.jfif')

# Function to load and preprocess an image

# Load the trained model
model_path = "model/model.h5"
model = load_model(model_path)
lr_model = pickle.load(open('model/lr_model.pkl', 'rb'))

    
selected = option_menu(
            menu_title=None, 
            options=["HOME","VERIFY","FUTURE SCOPE"],  
            orientation="horizontal",
        )

def plot_confusion_matrix(y_true, y_pred, categories):
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

def verify():

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Load and preprocess the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)
        sample_img = load_img(uploaded_file, target_size=(69, 69))
        sample_img_array = img_to_array(sample_img)
        sample_img_array = np.expand_dims(sample_img_array, axis=0)
        sample_img_features = model.predict(sample_img_array)
        sample_img_features_flat = sample_img_features.reshape(1, -1)
        sample_pred = int(np.round((lr_model.predict(sample_img_features_flat) + lr_model.predict(sample_img_features_flat)) / 2)[0])
        predicted_class_label= label_encoder.classes_[sample_pred]
        st.write(f"<p style='font-family: Times new roman; color: black; font-size: 20px;'>Predicted class: {predicted_class_label}</p>", unsafe_allow_html=True)
        if predicted_class_label=="sigatoka":
            describe = '<center><p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">Description</p></center>'
            st.markdown(describe, unsafe_allow_html=True)
            content = '<center><p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">Sigatoka is a foliar disease of banana caused by the fungus Pseudocercospora fijiensis.It is a fungal disease that destroys banana leaves, reducing the number and size of fruit.To avoid this removal of older leaves to reduce inoculum levels in a plantation, interplanting with other nonsusceptible crops, and planting in partial shade which results in less severe disease development.Timorex Gold is widely used for control the disease.</p></center>'
            st.markdown(content, unsafe_allow_html=True)
        elif predicted_class_label=="pestalotiopsis":
            describe = '<center><p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">Description</p></center>'
            st.markdown(describe, unsafe_allow_html=True)
            content = '<center><p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">Leaf spots will begin as very small, yellow, brown or black spots that enlarge in size. The spot usually turns gray with a black outline. Lesions on the petiole and rachis are similar. Symptoms may occur on multiple leaves at once, especially on juvenile palms.Spraying fungicides can be taken as preventive measures to control the spread of this disease.prochloraz and chlorothalonil is used to treat the diseases.</p></center>'
            st.markdown(content, unsafe_allow_html=True)
        elif predicted_class_label=="cordana":
            describe = '<center><p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">Description</p></center>'
            st.markdown(describe, unsafe_allow_html=True)
            content = '<center><p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">Typical symptoms of this disease appear as oval, pale brown spots, and as long strips of light brown necrosed tissue sometimes extending from the leaf margins to the midrib.Remove thatch at regular intervals and apply adequate nitrogen to help prevent the development of this disease.Control management of Cordana leaf spots in abaca involves the use of fungicides. In-vitro efficacy studies have shown that captasul and ridomil gold fungicides are highly toxic to the pathogen Cordana musae.</p></center>'
            st.markdown(content, unsafe_allow_html=True)
        else:
            describe = '<center><p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">Description</p></center>'
            st.markdown(describe, unsafe_allow_html=True)
            content = '<center><p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">Add plenty of well-composted manures or compost at planting time. Improve the soil in the planting area, at least three to four times the pot width and twice as deep as the pot. Apply a quality controlled release fertiliser. Mulch heavily but ensure it is well clear of the stem</p></center>'
            st.markdown(content, unsafe_allow_html=True)
            
def app():
    
    original_title = '<b><center><p style="font-family:Times new roman; color:black; font-size: 40px;">ABSTRACT</p></center></b>'
    st.markdown(original_title, unsafe_allow_html=True)
    content_1 = '<p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">Numerous disorders can impact the skin, which is the largest organ in the body and serves as a barrier against heat, light, injury, and infection. Nonetheless, appropriate treatment can result from a correct diagnosis. Early detection of skin illnesses is necessary to stop the growth and spread of skin lesions. There is a big reliance on the medical field. On the subject of information technology, a mechanism that can more accurately and quickly identify skin illnesses in their early stages is required in this day and age. Here try to implement the more accurate skin cancer classification with add one more step for preprocessing like segmentation and color correction for more accurate pixel values. Next main step is feature extraction because of the cancer cell classification know about the features of each so far use some steps to extract the features like LBP and PCA techniques. After feature extraction classification of the classes of the cancer cell by applying the deep learning algorithms such as CNN with more accurate layers and then also apply the machine learning algorithms for the classification. Finally performance metrics like accuracy, loss confusion matrix and classification report is displayed and comparison graph for the two algorithms also displayed.</p>'
    st.markdown(content_1, unsafe_allow_html=True)

def FUTURE():
     original_title = '<b><center><p style="font-family:Times new roman; color:black; font-size: 40px;">FUTURE SCOPE</p></center></b>'
     st.markdown(original_title, unsafe_allow_html=True)
     content_1 = '<p style="font-family:Times new roman; color:black; font-size: 20px; text-align: justify">The future scope of a comprehensive joint learning system for skin cancer detection lies in the integration of diverse data modalities, including clinical images, dermoscopic images, histopathology data, and patient records, to enhance the accuracy and reliability of skin cancer detection. Advancements in automated lesion segmentation, three-dimensional imaging, and the integration of genomic and molecular data offer the potential for a more nuanced understanding of skin lesions and their underlying biology. The development of explainable AI models will facilitate collaboration between the system and healthcare professionals, fostering trust and transparency in decision-making. Real-time decision support tools, seamless integration with imaging devices, and continuous model improvement mechanisms are key directions for ensuring the systems practicality and effectiveness in clinical settings. Ethical considerations, patient education tools, and participation in large-scale collaborative studies contribute to the responsible deployment and validation of the system on a global scale, ultimately advancing early detection and intervention in skin cancer.</p>'
     st.markdown(content_1, unsafe_allow_html=True)

def main():

    if selected == 'HOME':
        app()
    elif selected == 'VERIFY':
        verify()
    elif selected == 'FUTURE SCOPE':
        FUTURE()

if __name__ == "__main__":
    main()
