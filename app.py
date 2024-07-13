import streamlit as st
from typing import Dict
import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
from streamlit_option_menu import option_menu
from torchvision import transforms
from model import ResNet50
from torchvision.transforms.functional import to_pil_image
import webbrowser
import base64
import io

def get_file_content_as_string(path_1):
    path = os.path.dirname(__file__)
    my_file = path + path_1
    with open(my_file,'r') as f:
        instructions=f.read()
    return instructions

def article_text(path):
    with open(path, 'r') as file:
        readme_text = file.read()

    readme_html = f"<div style='text-align: justify'>{readme_text}</div>"
    st.markdown(readme_html, unsafe_allow_html=True)

def home():
    st.write('''# Garbage Classification''')

    article_text('notepad/Instructions.md')

    tab1, tab2 = st.tabs(["Kategori 1", "Kategori 2"])

    with tab1:
        st.image('image/3 tempah sampah.jpg', caption='Sumber: https://mmc.kalteng.go.id/berita/read/1868/empat-jenis-tempat-sampah-yang-perlu-diketahui')
        article_text('notepad/kategori_1.md')
    with tab2:
        st.image('image/5 tempat sampah.jpg', caption='Sumber: https://bobo.grid.id/read/08679812/mengenal-warna-tempat-sampah-yuk?page=all')
        article_text('notepad/kategori_2.md')

device = torch.device('cpu')

def load_model():
    model = ResNet50(13).to(device)
    weights = torch.load('resnet50_models_fixed.pth', map_location=device)

    model.load_state_dict(weights)
    
    model.eval()
    return model 

model = load_model()
label2cat  = ['battery', 'cigarattes', 'clothes', 'fruits', 
              'glass', 'lamp', 'meat', 'medical waste', 
              'metal', 'paper', 'plastic', 'rice', 'vegetables']

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
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
            'nama': 'Buah',
            'image': './image/tong hijau.png',
            'warna': 'Hijau',
            'unsafe_allow_html': True
        },
        'meat': {
            'content': './notepad/organik/meat.md',
            'tipe': 'Organik',
            'nama': 'Daging',
            'image': './image/tong hijau.png',
            'warna': 'Hijau',
            'unsafe_allow_html': True,
        },
        'paper': {
            'content': './notepad/organik/paper.md',
            'tipe': 'Organik',
            'nama': 'Kertas',
            'image': './image/tong hijau.png',
            'warna': 'Hijau',
            'unsafe_allow_html': True,
        },
        'rice': {
            'content': './notepad/organik/rice.md',
            'tipe': 'Organik',
            'nama': 'Nasi',
            'image': './image/tong hijau.png',
            'warna': 'Hijau',
            'unsafe_allow_html': True,
        },
        'vegetables': {
            'content': './notepad/organik/vegetables.md',
            'tipe': 'Organik',
            'nama': 'Sayur',
            'image': './image/tong hijau.png',
            'warna': 'Hijau',
            'unsafe_allow_html': True,
        },
        'clothes': {
            'content': './notepad/anorganik/clothes.md',
            'tipe': 'Anorganik',
            'nama': 'Baju',
            'image': './image/tong kuning.png',
            'warna': 'Kuning',
            'unsafe_allow_html': True,
        },
        'glass': {
            'content': './notepad/anorganik/glass.md',
            'tipe': 'Anorganik',
            'nama': 'Kaca',
            'unsafe_allow_html': True,
        },
        'metal': {
            'content': './notepad/anorganik/metal.md',
            'tipe': 'Anorganik',
            'nama': 'Metal',
            'image': './image/tong kuning.png',
            'warna': 'Kuning',
            'unsafe_allow_html': True,
        },
        'plastic': {
            'content': './notepad/anorganik/plastic.md',
            'tipe': 'Anorganik',
            'nama': 'Plastik',
            'image': './image/tong kuning.png',
            'warna': 'Kuning',
            'unsafe_allow_html': True,
        },
        'battery': {
            'content': './notepad/b3/battery.md',
            'tipe': 'B3',
            'nama': 'Baterai',
            'image': './image/tong merah.png',
            'warna': 'Merah',
            'unsafe_allow_html': True,
        },
        'cigarattes': {
            'content': './notepad/b3/cigarattes.md',
            'tipe': 'B3',
            'nama': 'Rokok',
            'image': './image/tong merah.png',
            'warna': 'Merah',
            'unsafe_allow_html': True,
        },
        'lamp': {
            'content': './notepad/b3/lamp.md',
            'tipe': 'B3',
            'nama': 'Lampu',
            'image': './image/tong merah.png',
            'warna': 'Merah',
            'unsafe_allow_html': True,
        },
        'medical waste': {
            'content': './notepad/b3/medical_waste.md',
            'tipe': 'B3',
            'nama': 'Limbah Medis',
            'image': './image/tong merah.png',
            'warna': 'Merah',
            'unsafe_allow_html': True,
        },
    }

    content = category_content_maps[category]

    col1, col2 = st.columns([1, 2])
  
    with col1:
        image = Image.open(content['image']).convert('RGB')

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
            f"""
            <style>
            .image-wrapper {{
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .image-wrapper img {{
                max-width: 100%;
                max-height: 100%;
            }}
            </style>

            <div class="image-wrapper">
                <img src="data:image/png;base64,{image_data}" alt="Gambar">
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.write('')
        st.info(f'Jenis sampah ini dibuang pada tempat sampah warna {content["warna"]}')

        st.markdown(
            f'<div style="display: flex; margin-top: 10px; margin-bottom: 70px; justify-content: center;"><span style="background-color: green; color: white; padding: 10px 50px; font-size: 20px;">{content["nama"]}: {content["tipe"]}</span></div>',
            unsafe_allow_html=True
        )
    
    
    with open(content['content'], 'r') as file:
        readme_text = file.read()

    readme_html = f"""<div style='text-align: justify'>{readme_text}</div>"""
    st.markdown(readme_html, unsafe_allow_html=True)
   
def content2(category):
    category_content_maps = {
        'fruits': {
            'content': './notepad/organik/fruits.md',
            'tipe': 'Sampah Organik',
            'nama': 'Buah',
            'image': './image/tong hijau.png',
            'warna': 'Hijau',
            'unsafe_allow_html': True
        },
        'meat': {
            'content': './notepad/organik/meat.md',
            'tipe': 'Sampah Organik',
            'nama': 'Daging',
            'image': './image/tong hijau.png',
            'warna': 'Hijau',
            'unsafe_allow_html': True,
        },
        'rice': {
            'content': './notepad/organik/rice.md',
            'tipe': 'Sampah Organik',
            'nama': 'Nasi',
            'image': './image/tong hijau.png',
            'warna': 'Hijau',
            'unsafe_allow_html': True,
        },
        'vegetables': {
            'content': './notepad/organik/vegetables.md',
            'tipe': 'Sampah Organik',
            'nama': 'Sayur',
            'image': './image/tong hijau.png',
            'warna': 'Hijau',
            'unsafe_allow_html': True,
        },
        'glass': {
            'content': './notepad/anorganik/glass.md',
            'tipe': 'Sampah Guna Ulang',
            'nama': 'Kaca',
            'image': './image/tong kuning.png',
            'warna': 'Kuning',
            'unsafe_allow_html': True,
        },
        'metal': {
            'content': './notepad/anorganik/metal.md',
            'tipe': 'Sampah Guna Ulang',
            'nama': 'Metal',
            'image': './image/tong kuning.png',
            'warna': 'Kuning',
            'unsafe_allow_html': True,
        },
        'plastic': {
            'content': './notepad/anorganik/plastic.md',
            'tipe': 'Sampah Guna Ulang',
            'nama': 'Plastik',
            'image': './image/tong kuning.png',
            'warna': 'Kuning',
            'unsafe_allow_html': True,
        },
        'battery': {
            'content': './notepad/b3/battery.md',
            'tipe': 'Sampah B3',
            'nama': 'Baterai',
            'image': './image/tong merah.png',
            'warna': 'Merah',
            'unsafe_allow_html': True,
        },
        'lamp': {
            'content': './notepad/b3/lamp.md',
            'tipe': 'Sampah B3',
            'nama': 'Lampu',
            'image': './image/tong merah.png',
            'warna': 'Merah',
            'unsafe_allow_html': True,
        },
        'medical waste': {
            'content': './notepad/b3/medical_waste.md',
            'tipe': 'Sampah B3',
            'nama': 'Limbah Medis',
            'image': './image/tong merah.png',
            'warna': 'Merah',
            'unsafe_allow_html': True,
        },
        'paper': {
            'content': './notepad/organik/paper.md',
            'tipe': 'Sampah Daur Ulang',
            'nama': 'Kertas',
            'image': './image/tong biru.png',
            'warna': 'Biru',
            'unsafe_allow_html': True,
        },
        'clothes': {
            'content': './notepad/anorganik/clothes.md',
            'tipe': 'Sampah Guna Ulang',
            'nama': 'Baju',
            'image': './image/tong kuning.png',
            'warna': 'Kuning',
            'unsafe_allow_html': True,
        },
        'cigarattes': {
            'content': './notepad/b3/cigarattes.md',
            'tipe': 'Sampah Residu',
            'nama': 'Rokok',
            'image': './image/tong abu.png', 
            'warna': 'Abu-abu',
            'unsafe_allow_html': True,
        },
    }

    content = category_content_maps[category]

    col1, col2 = st.columns([1, 2])
  
    with col1:
        image = Image.open(content['image']).convert('RGB')

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(
            f"""
            <style>
            .image-wrapper {{
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .image-wrapper img {{
                max-width: 100%;
                max-height: 100%;
            }}
            </style>

            <div class="image-wrapper">
                <img src="data:image/png;base64,{image_data}" alt="Gambar">
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.write('')
        st.info(f'Jenis sampah ini dibuang pada tempat sampah warna {content["warna"]}')

        st.markdown(
            f'<div style="display: flex; margin-top: 10px; margin-bottom: 70px; justify-content: center;"><span style="background-color: green; color: white; padding: 10px 50px; font-size: 20px;">{content["nama"]}: {content["tipe"]}</span></div>',
            unsafe_allow_html=True
        )
    
    with open(content['content'], 'r') as file:
        readme_text = file.read()

    readme_html = f"""<div style='text-align: justify'>{readme_text}</div>"""
    st.markdown(readme_html, unsafe_allow_html=True)

def main_clf(option, content_func):
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
                    st.markdown('#')
                    if st.button('Classify'): 
                        category = predict_image(image)

                st.write('-'*3)

                if len(category) > 0:
                    content_func(category)
                    
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
            content_func(category)

def page_2():
    st.write('''# Garbage Classification''')
    # st.markdown('### Choose your preferred method')
    
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    select_box = st.selectbox(
        'Pick Classification by', ('', 'Kategori 1', 'Kategori 2'))
    
    if select_box == 'Kategori 1':
        st.success('You selected Kategori 1')
        option = st.radio('### Choose your option', ['Upload a image', 'Take a photo'])
        main_clf(option, content)
    elif select_box == 'Kategori 2':
        st.success('You selected Kategori 2')
        option = st.radio('### Choose your option', ['Upload a image', 'Take a photo'])
        main_clf(option, content2)
     
def page_3():
    st.write('''# About''')
    st.text('')

    with open('./notepad/about.md', 'r') as file:
        readme_text = file.read()

    readme_html = f"<div style='text-align: justify'>{readme_text}</div>"
    readme_markdown = st.markdown(readme_html, unsafe_allow_html=True)

def page_4():
    with open('./notepad/menu_help.md', 'r') as file:
        readme_text = file.read()

    readme_html = f"<div style='text-align: justify'>{readme_text}</div>"
    readme_markdown = st.markdown(readme_html, unsafe_allow_html=True)


def main():
    with st.sidebar:
        page = option_menu('Select a page', ['Home', 'Classification', 'About', 'Help'], default_index=0)
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




