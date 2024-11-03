import streamlit as st
import joblib

def main():
  st.title("BSI Mobile - Uji Ulasan Baru")

  # logo
  st.logo("img/unimal_logo.png")

  # TESTING
  # input data testing
  st.header("Input Ulasan Baru")

  # Input ulasan baru dari pengguna
  input_text = st.text_input("Masukkan Ulasan Baru: ")

  # Memuat model yang sudah disimpan
  loaded_model = joblib.load('model/model_bsi.pkl')
        
  if input_text:
    # Prediksi hasil analisis menggunakan model grid_search
    result_test = loaded_model.predict([input_text])
    # Tampilkan hasil analisis
    st.write(f"Hasil Analisis: {result_test[0]}")
  # END 


if __name__ == "__main__":
    main()

# Footer
st.markdown("""
---
Made with ❤️ in Lhokseumawe by *Mr Mustache* 🥸. All rights reserved.
""")