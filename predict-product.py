import streamlit as st
import numpy as np
import joblib

st.title('Predict Product📈')
st.write('Masukkan harga, jumlah review dan rating product. Sistem akan memberi prediksi jumlah penjualan produk😆')

input = st.text_input('Harga💲')
input2 = st.text_input('Jumlah Review Produk💬')
input3 = st.text_input('Rating Produk⭐')

# Load the model from the file, Model is build use Decission Tree with MAE value 40.08 
model = joblib.load('Model.h5')

if(st.button('Predict')):
    input_user = [input, input2, input3]
    input_user = np.array(input_user, dtype=np.float64)
    input_user = input_user.reshape(1, -1)
    prediksi = model.predict(input_user)[0]
    
    st.write(f'**Hasil Prediksi penjualan produk adalah {prediksi} ✨**')