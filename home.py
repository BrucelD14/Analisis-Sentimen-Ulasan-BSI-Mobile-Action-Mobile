# import package necessary
import streamlit as st

# Set the title of the page
st.set_page_config(page_title="Web Analisis Sentimen", page_icon="🤖")

# MAIN FUNCTION
def main():
    # Title of the landing page
    st.title("Analisis Sentimen Ulasan Pengguna Aplikasi BSI Mobile Dan Action Mobile")

    # logo
    st.logo("img/unimal_logo.png") 

    # Introduction section
    st.write("""
    Halo, selamat datang di website **ANALISIS SENTIMEN APLIKASI BSI MOBILE DAN ACTION MOBILE**. Website ini adalah sebuah sistem yang dirancang guna meng-evaluasi tingkat kepuasan pengguna terhadap layanan aplikasi mobile banking yang disediakan oleh BANK BSI dan BANK ACEH. Hasil sistem ini akan mengklasifikasikan ulasan pengguna ke dalam kategori Positif dan kategori Negatif berdasarkan ulasan yang dimuat pada Google Play Store.
    """)

    # Information about the applications
    st.header("Detail Informasi Mobile Banking")
    st.write("""
    Mobile banking merupakan layanan bank digital yang dapat mempermudah nasabah melakukan transaksi, sebagai bentuk adaptasi pasar Bank Syariah Indonesia menyediakan layanan BSI Mobile dan Bank Aceh menyediakan layanan Action Mobile. 
    - **BSI MOBILE** : BSI Mobile merupakan layanan mobile banking yang disediakan Bank Syariah Indonesia dengan memuat beragam fitur yang dapat digunakan dan mempermudah nasabahnya dalam melakukan berbagai jenis transaksi.
    - **ACTION MOBILE** : Action Mobile merupakan layanan mobile banking yang disediakan oleh Bank Aceh. Bank Aceh merupakan bank daerah yang hanya terdapat pada provinsi Aceh.
    """)

    # Features of the sentiment analysis tool
    st.header("Detail Sistem")
    st.write("""
    Sistem ini akan memuat beberapa detail informasi mengenai hasil proses analisis sentimen yang dilakukan, hal tersebut berupa:
    - **Preprocessing Dataset** 🧮 : Preprocessing menampilkan data yang telah sesuai dengan standarisasi text pemahaman sistem.
    - **Klasifikasi Sentimen** 🔎 : Sistem ini akan menampilkan ulasan penggun dengan kategori sentimen *Positif* atau *Negatif*.
    - **Visualisasi Sentimen** 📊 : Visualisasi yang ditampilkan akan berupa WordCloud dengan representasi kata terbanyak pada setiap kelas sentimen positif dan negatif.
    - **Evaluasi Model** 📈 : Evaluasi Model akan menampilkan tingkat akurasi algoritma Multinomial Naive Bayes yang digunakan pada proses analisis.
    """)

    # CTA section
    st.header("*Get Started*")
    st.write("Untuk mulai menganalisis sentimen ulasan pengguna untuk BSI Mobile dan Action Mobile, gunakan menu navigasi di sebelah kiri. ⬅️")


if __name__ == "__main__":
    main()


# Footer
st.markdown("""
---
Made with ❤️ in Lhokseumawe by *Brucel Duta* 👾. All rights reserved.
""")


