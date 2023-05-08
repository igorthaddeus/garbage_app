def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

     st.markdown("""
                    <h1 style='text-align: center; color: grey;'>
                        Big headline
                    </h1>""", 
                    unsafe_allow_html=True)

# st.write('Help.')
# # Store the initial value of widgets in session state
# if "horizontal" not in st.session_state:
#     st.session_state.horizontal = True

# st.radio(
#     "Set label visibility 👇",
#     ["aaa", "bbb", "ccc"],
#     label_visibility="visible",
#     disabled=False,
#     horizontal=st.session_state.horizontal,
# )


# Define the navbar function
# def navbar(links):
#     nav_html = '''
#     <style>
#         .navbar {
#             display: flex;
#             flex-wrap: wrap;
#             justify-content: space-between;
#             align-items: center;
#             padding: 0.5rem 1rem;
#             background-color: #ffffff;
#             color: #000000;
#             border-bottom: 1px solid #cccccc;
#         }
#         .nav-link {
#             margin: 0.5rem;
#             font-size: 1.2rem;
#             text-decoration: none;
#             color: #000000;
#         }
#     </style>
#     <nav class='navbar'>
#         <div>
#             <span class='nav-link'>
#                 <b>Sampah Classification</b>
#             </span>
#         </div>
#         <div>
#             '''
#     for text, link in links.items():
#         nav_html += f'<a class='nav-link' href='{link}'>{text}</a>'
#     nav_html += '''
#         </div>
#     </nav>
#     '''
#     return nav_html

# Set up the navigation links

nav_links = {
    'Home': '/',
    'Page 2': '/page-2',
    'Page 3': '/page-3'
}

# if file is None:
#         st.text("Please upload an image file")
#     else:
#         image = Image.open(file)
#         st.image(image, use_column_width=True)

#         image = Image.open(file)
#         image = image.resize((400,400)) # ubah ukuran gambar menjadi 400x400
#         st.image(image, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB', output_format='auto', style='display: block; margin: auto;')
#         image_bytes = image.tobytes()
#         base64_bytes = base64.b64encode(image_bytes).decode('utf-8')
#         data_uri = f'data:image/jpg;base64,{base64_bytes}'

#         # Display the image in the center of the page
#         st.markdown(f'<p style="text-align:center;"><img src="{data_uri}"></p>', unsafe_allow_html=True)

#         predictions = import_and_predict(image, model)
#         score = tf.nn.softmax(predictions[0])
#         st.write(prediction)
#         st.write(score)
#         print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(class_names[np.argmax(score)], 100 * np.max(score)))



# <p style="display: flex; justify-content: center; bottom:0; left:0; padding:1px 0; width:100%; padding-bottom: 10px; background-color:#fff; color:#000;">
#         Source: https://www.vecteezy.com/vector-art/7307160-go-green-cute-dustbin-container-and-planet-earth-no-plastic-go-green-lettering-poster-t-shirt-textile-graphic-design-beautiful-illustration-protest-against-plastic-garbage
#     </p>