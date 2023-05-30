import streamlit as st
import cv2
import numpy as np
import base64
import os
import torch
from PIL import Image, ImageOps
from streamlit_option_menu import option_menu
from torchvision import transforms
from torchvision.models import mobilenet_v2
from model import ResNet50
import io
from torchvision.transforms.functional import to_pil_image

# Read File
def get_file_content_as_string(path_1):
    path = os.path.dirname(__file__)
    my_file = path + path_1
    with open(my_file,'r') as f:
        instructions=f.read()
    return instructions

# Define the home page
def home():
    st.write('''# Garbage Classification''')

    # image = st.image('go-green.png',
    #                  caption='Source: https://www.kaggle.com/c/cdiscount-image-classification-challenge/overview', width=200)
    
    # st.markdown(
    #     f'<div style="display: flex; justify-content: center;"><img src="data:image/*;base64,{encoded_string}" width="300"/></div>',
    #     unsafe_allow_html=True
    # )

    with open("./image/go-green.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    st.markdown(f"""
                <div style="position:relative;">
                    <div style="display: flex; justify-content: center;">
                        <img src="data:image/*;base64,{encoded_string}" width="300"/>
                    </div>
                </div>""", unsafe_allow_html=True)
    
    # readme_text = st.markdown(get_file_content_as_string('./notepad/Instructions.md'), unsafe_allow_html=True)

    with open('./notepad/Instructions.md', 'r') as file:
        readme_text = file.read()

    readme_html = f"<div style='text-align: justify'>{readme_text}</div>"
    readme_markdown = st.markdown(readme_html, unsafe_allow_html=True)


device = torch.device('cpu')

def load_model():
    model = ResNet50(13).to(device)
    # weights = torch.load('model/resnet50-state_dict.pth', map_location=device)
    weights = torch.load('model/resnet50_v2.pth', map_location=device)
    
    model.load_state_dict(weights)
    model.eval()
    return model 

model = load_model()
label2cat  = ['battery', 'cigarattes', 'clothes', 'fruits', 
              'glass', 'lamp', 'meat', 'medical waste', 
              'metal', 'paper', 'plastic', 'rice', 'vegetables']

def predict_image(image_path):
    # image_path = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()  
    with torch.no_grad():
        output = model(image_tensor)
    prob = torch.softmax(output, dim=1)
    pred_class_idx = prob.argmax(dim=1)
    prediction = label2cat[pred_class_idx]

    print(pred_class_idx)
    print(f'Prediction: {prediction}')
    return prediction

# CSS
st.markdown('''
    <style>
        div.stButton > button:first-child {
            background-color: #05c4bc;
            border-radius: 10px;
            color: #fff;
            padding: 5px 20px;
            justify-content: center;
            align-items: center;
            display: flex;
            margin: 0 auto;
    }
    </style>''', unsafe_allow_html=True)

st.markdown("""
    <style>
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 150px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

def content(category):
    category_content_maps = {
        'fruits': {
            'content': './notepad/organik/fruits.md',
            'tipe': 'Organik',
            'unsafe_allow_html': True
        },
        'meat': {
            'content': './notepad/organik/meat.md',
            'tipe': 'Organik',
            'unsafe_allow_html': True,
        },
        'paper': {
            'content': './notepad/organik/paper.md',
            'tipe': 'Organik',
            'unsafe_allow_html': True,
        },
        'rice': {
            'content': './notepad/organik/rice.md',
            'tipe': 'Organik',
            'unsafe_allow_html': True,
        },
        'vegetables': {
            'content': './notepad/organik/vegetables.md',
            'tipe': 'Organik',
            'unsafe_allow_html': True,
        },
        'clothes': {
            'content': './notepad/anorganik/clothes.md',
            'tipe': 'Anorganik',
            'unsafe_allow_html': True,
        },
        'glass': {
            'content': './notepad/anorganik/glass.md',
            'tipe': 'Anorganik',
            'unsafe_allow_html': True,
        },
        'metal': {
            'content': './notepad/anorganik/metal.md',
            'tipe': 'Anorganik',
            'unsafe_allow_html': True,
        },
        'plastic': {
            'content': './notepad/anorganik/plastic.md',
            'tipe': 'Anorganik',
            'unsafe_allow_html': True,
        },
        'battery': {
            'content': './notepad/b3/battery.md',
            'tipe': 'B3',
            'unsafe_allow_html': True,
        },
        'cigarattes': {
            'content': './notepad/b3/cigarattes.md',
            'tipe': 'B3',
            'unsafe_allow_html': True,
        },
        'lamp': {
            'content': './notepad/b3/lamp.md',
            'tipe': 'B3',
            'unsafe_allow_html': True,
        },
        'medical waste': {
            'content': './notepad/b3/medical_waste.md',
            'tipe': 'B3',
            'unsafe_allow_html': True,
        },
        # 'medical_waste': {
        #     'content': './notepad/b3/medical_waste.md',
        #     'tipe': 'B3',
        #     'unsafe_allow_html': True,
        # },
    }

    content = category_content_maps[category]
    # markdown = st.markdown(f"<h2 style='text-align: center; color: black;'>{category}</h2>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="display: flex; margin-top: 23px;justify-content: center;"><span style="background-color: green; color: white; padding: 10px 20px;">{category}: {content["tipe"]}</span></div>',
        unsafe_allow_html=True
    )
    # st.success(f'{category}, {content["tipe"]}')

    with open(content['content'], 'r') as file:
        readme_text = file.read()

    readme_html = f"""<div style='text-align: justify'>{readme_text}</div>"""
    st.markdown(readme_html, unsafe_allow_html=True)

# Define page 2
def page_2():
    st.write('''# Garbage Classification''')

    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.markdown('### Choose your preferred method')
    option = st.radio('### Choose your option', ['Upload a image', 'Take a photo'])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if option == 'Upload a image':
        uploaded_file = st.file_uploader('Choose an image')
        category = ''
        if uploaded_file is not None:
            try:
                col1, col2 = st.columns([1,1])

                with col1:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, width=300)
                
                with col2:
                    st.success('Successfully uploaded the image. Please see the category predictions below')
                    st.warning('To get a classify result, press the Classify buttton!')
                    st.write('-'*3)

                    if st.button('Classify'): 
                        category = predict_image(image)

                if len(category) > 0:
                    content(category)
                    
            except Exception as e:
                st.error('Please upload an image in jpg or png format..!')
                st.error(f'Error: {e}')
        else:
            st.info('Input image')

    if option == 'Take a photo':
        picture = st.camera_input("Take a picture")

        if picture:
            bytes_data = picture.getvalue()
            torch_img = torch.ops.image.decode_image(
                torch.from_numpy(np.frombuffer(bytes_data, np.uint8)), 3
            )
            pil_image = to_pil_image(torch_img)
            category = predict_image(pil_image)
            st.success(f'Predicted category: {category}')
            content(category)

# Define page 3
def page_3():
    st.write('''# About''')
    st.text('')

    # with open("./image/yuuki yoda.png", "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # st.markdown(
    #     f'<div style="display: flex; justify-content: center;"><img src="data:image/*;base64,{encoded_string}" width="300"/></div>',
    #     unsafe_allow_html=True
    # )

    readme_text = st.markdown(get_file_content_as_string('./notepad/about.md'), unsafe_allow_html=True)

def page_4():
    readme_text = st.markdown(get_file_content_as_string('./notepad/menu_help.md'), unsafe_allow_html=True)
    

# Create the app
def main():
    # st.set_page_config(page_title='Sampah Classification', page_icon=':recycle:')
    # st.markdown(navbar(nav_links), unsafe_allow_html=True)
    with st.sidebar:
        page = option_menu('Select a page', ['Home', 'Classification', 'About', 'Help'],
                           default_index=0)
        
    if page == 'Home':
        home()
    elif page == 'Classification':
        page_2()
    elif page == 'About':
        page_3()
    elif page == 'Help':
        page_4()

if __name__ == '__main__':
    main()
