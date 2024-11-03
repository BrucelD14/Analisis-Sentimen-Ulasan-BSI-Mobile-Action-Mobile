# Importing Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.probability import FreqDist


nltk.download('punkt')
nltk.download('stopwords')
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud, STOPWORDS
import re
import joblib

def main():
    st.title("BSI Mobile - Analisis Sentimen")

    # logo
    st.logo("img/unimal_logo.png") 

    st.write("""
    **Features:**
    - Preprocessing Dataset 🧮
    - Klasifikasi Sentimen 🔎
    - Visualisasi Sentimen 📊
    - Evaluasi Model 📈
    """)
    # Add more content related to BSI Mobile here
    # Judul aplikasi
    st.header("Form Input File CSV")

    # Membuat form untuk mengunggah file CSV
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])

    if uploaded_file is not None:
        # Membaca file CSV menjadi DataFrame
        df = pd.read_csv(uploaded_file)

        # Menampilkan DataFrame
        st.write("Berikut adalah data dari file CSV yang Anda unggah:")
        st.dataframe(df.head())

        # Menampilkan informasi tambahan (opsional)
        # st.write("Informasi tentang DataFrame:")
        # st.write(df.describe())

        # PREPROSESING
        st.header("Cleansing Dataset")
        st.subheader("Check data shape & Drop Data Duplicate")
        # Check data shape
        st.write(f"Data has: {df.shape[0]} rows and {df.shape[1]} columns")

        # drop data duplicate
        df = df.drop_duplicates(subset='content')
        st.write(f"Data after removing duplicates: {df.shape[0]} rows and {df.shape[1]} columns")

        # Removing Null
        st.subheader("Removing Null")
        df = df.dropna()
        st.write(f"Data after removing null: {df.shape[0]} rows and {df.shape[1]} columns")

        st.subheader("Menghapus simbol dan angka")
        # menghapus simbol, angka
        def clean_bsi_mobile_data(text):
            text = re.sub(r'@[A-Za-z0-9_]+', '', text)
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'RT[\s]+', '', text)
            text = re.sub(r'https?://\S+', '', text)
            
            text = re.sub(r'[^A-Za-z0-9 ]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text

        df['content'] = df['content'].apply(clean_bsi_mobile_data)

        # Function to remove numbers from text
        def remove_numbers(text):
            return re.sub(r'\d+', '', text)

        # Apply the function to the 'content' column
        df['content'] = df['content'].apply(remove_numbers)
        st.dataframe(df.head())

        st.subheader("Mengubah menjadi lower case")
        # To Lower Case
        df['content'] = df['content'].str.lower()
        st.dataframe(df.head())

        st.subheader("Filter baris di mana jumlah kata dalam kolom 'content' kurang dari 4")
        # Langkah 3: Filter baris di mana jumlah kata dalam kolom 'content' kurang dari 4
        df = df[df['content'].apply(lambda x: len(x.split()) >= 4)]
        st.dataframe(df.head())

        st.header("Data Preprosesing")

        st.subheader("Normalisasi Kata")
        # Normalisasi
        norm = {' gk ' : ' nggak ', ' mmg ' : ' memang ', ' krna ' : ' karena ', ' krn ' : ' karena ',
                ' no ' : ' nomor ', ' ktp ' : ' kartu tanda penduduk ', ' trus ' : ' terus ',
                ' yobain ' : ' coba ', ' sya ' : ' saya ', ' doang ' : ' saja ', ' kaga ' : ' nggak ', 
                ' mf ' : ' maaf ', ' sayah ' : ' saya ', ' ilang ' : ' hilang ', ' apk ' : ' aplikasi ', 
                ' gua ' : ' saya ', ' duwit ' : ' uang ', ' yg ' : ' yang ', ' gx ' : ' nggak ', 
                    ' knpa ' : ' kenapa ', ' skrng ' : ' sekarang ', ' kyk ' : ' seperti ', 
                ' bgt ' : ' sangat ', ' naroh ' : ' meletakkan ', ' mw ' : ' mau ', ' sdh ' : ' sudah ', 
                ' dapt ' : ' dapat ', ' bukak ' : ' buka ', ' tdk ' : ' tidak ', ' jg ' : ' juga ',
                ' makasih ' : ' terimakasih ', ' makin ' : ' semakin ', ' ga ': ' nggak ', ' ngirim ' : ' mengirim ',
                ' knp ' : ' kenapa ', ' muter ' : ' putar ', ' ni ' : ' ini ', ' skarang ' : ' sekarang ', 
                ' kalo ' : ' kalau ', ' jgn ' : ' jangan ', ' bgtu ' : ' begitu ',  
                ' bales ' : ' balas ', ' blm ' : ' belum ', ' bgs ' : ' bagus ', ' cmn ' : ' cuma ', 
                ' dah ' : ' sudah ', ' mnding ' : ' mending ', ' pdhal ' : ' padahal ', ' smua ' : ' semua ', 
                ' lg ' : ' lagi ', ' dri ' : ' dari ', ' tida ' : ' tidak ', ' nmr ' : ' nomor ', ' br ' : ' baru ', 
                ' tmn ' : ' teman ', ' gw ' : ' aku ', ' aja ' : ' saja ', 
                ' tlp ' : ' telepon ', ' sistim ' : ' sistem ', ' udh ' : ' sudah ', ' goblooook ' : ' bodoh ', 
                ' vermuk ' : ' verifikasi muka ', ' seyelah ' : ' setelah ', ' opresinal ' : ' operasional ', 
                    ' aktivitasi ' : ' aktivasi ', ' gabisa ' : ' nggak bisa ', ' garagara ' :' dikarenakan  ',
                ' trs ' : ' terus ', ' verivikasi ' : ' verifikasi ', ' tlg ' : ' tolong ', 
                ' moga ' : ' semoga ', ' ngga ' : ' tidak ', ' diem ' : ' diam ', ' klo ' : ' kalau ', 
                ' kayak ' : ' seperti ', ' tololll ' : ' bodoh ', ' ngak ' : ' nggak ', ' tpi ' : ' tetapi ',
                ' bengking ' : ' mobile banking ', ' jd ' : ' jadi ', ' bs ' : ' dapat ', ' g ' : ' nggak ', 
                ' ekspetasi ' : ' harapan ', ' ko ' : ' mengapa ', ' ajg ' : ' anjing ', ' kok ' : ' mengapa ', 
                ' trasaksi ' : ' transaksi ', ' utk ' : ' untuk ', ' berkalikali ' : ' berulang ', ' sampe ' : ' sampai ', 
                ' biar ' : ' agar ', ' dg ' : ' dengan ', ' gak ' : ' nggak ', ' pas ' : ' saat ', 
                ' perbankan ' : ' bank ', ' error ' : ' eror ', ' bikin ' : ' buat ', ' smoga ' : ' semoga ', 
                ' smg ' : ' semoga ', ' udah ' : ' sudah ', ' hp ' : ' smartphone ', ' login ' : ' masuk ', 
                ' uda ' : ' sudah ', ' bgt ' : ' banget ', ' ribet ' : ' sulit ', ' download ' : ' unduh ', 
                ' lgi ' : ' lagi ', ' mesti ' : ' harus ', ' nmr ' : ' nomor ', ' gimana ' : ' bagaimana ', 
                ' gmn ' : ' bagaimana ', ' nanya ' : ' tanya ', ' kagak ' : ' tidak ', 
                ' device ' : ' perangkat ', ' abis ' : ' habis ', ' sbg ' : ' sebagai ', ' bug ' : ' cacat ', 
                ' nampak ' : ' terlihat ', ' thn ' : ' tahun ', ' pakek ' : ' pakai ', ' lelet ' : ' lemot ', 
                ' cuman ' : ' hanya ', ' makasih ' : ' terimakasih', ' duit ' : ' uang ', ' tak ' : ' tidak ', 
                ' dah ' : ' sudah ', ' notif ' : ' notifikasi ', ' upgrade ' : ' tingkatkan ', 
                ' support ' : ' dukungan ', ' komen ' : ' komentar ', ' via ' : ' melalui ', ' gampang ' : ' mudah ',
                ' org ' : ' orang ', ' dmn ' : ' dimana ', ' blank ' : ' hilang ',
                ' jaringan ' : ' koneksi', ' sinyal ' : ' koneksi ', ' nih ' : ' ini ', ' seneng ' : ' senang ', 
                ' gini ' : ' begini ', ' gitu ' : ' begitu ', ' pake ' : ' pakai ', 
                ' ngga ' : ' nggak', ' info ' : ' informasi ', ' mantep ' : ' mantap ',
                ' karna ' : ' karena ', ' jdi ' : ' jadi ', ' good job ' : ' good', ' cepet ' : ' cepat ', 
                ' handphone ' : ' smartphone ', ' telepon ' : ' smartphone ', ' bener ' : ' benar ', ' dgn ' : ' dengan ', 
                ' simple ' : ' simpel ', ' liat ' : ' lihat ', ' mantab ' : ' mantap ', ' tf ' : ' transfer ', 
                ' cs ' : ' layanan nasabah ', ' uptodate ' : ' terkini ', ' updated ' : ' terbaru', 
                ' hadeh ' : ' aduh ', ' gamau ' : ' nggak mau ', ' perfect ' : ' sempurna ', 
                ' ayok ' : ' ayo ', ' yuk ' : ' ayo ', ' kuy ' : ' ayo ', ' best ' : ' terbaik ',  
                ' problem ' : ' masalah ', ' trouble ' : ' masalah ', ' debest ' : ' terbaik ', ' thanks ' : ' terimakasih ', 
                ' gapernah ' : ' nggak pernah ', ' thankyou ' : ' terimakasih ', ' engga ' : ' nggak ', ' bagu ' : ' bagus ', 
                ' stuck ' : ' berhenti ', ' bngt ' : ' banget ', ' ngk ' : ' nggak ', ' emng ' : ' memang ', ' emg ' : ' memang ', 
                ' payah ' : ' susah ', ' sulit ' : ' susah ', ' gaperlu ' : ' nggak perlu ', ' lbh ' : ' lebih ', 
                ' komplit ' : ' komplet ', ' dll ' : ' dan lain lain ', ' user ' : ' pengguna ', ' friendly ' : ' ramah ', 
                ' luv ' : ' cinta ', ' love ' : ' cinta ', ' kyk ' : ' seperti ', ' dlm ' : ' dalam ', ' mager ' : ' malas ', 
                ' gemes ' : ' gemas ', ' satset ' : ' cepat ', ' nice ' : ' bagus ', ' debes ' : ' terbaik ', ' jos ' : ' jaya ', 
                ' ilang ' : ' hilang ', ' mantul ' : ' mantap ', ' bkn' : ' bukan ', ' cape ' : ' capek ', 
                ' customer ' : ' nasabah ', ' kudu ' : ' harus ', ' goodjob ' : ' good', ' ttp ' : ' tetap ', 
                ' tetep ' : ' tetap ', ' alhasil ' : ' hasil ', ' mesjid ' : ' masjid ', ' tq ' : ' terimakasih ', 
                ' kasian ' : ' kasihan ', ' kecelek ' : ' kaget ', ' dr ' : ' dari ', ' skrg ' : ' sekarang ', 
                ' pokonya ' : ' pokok nya ', ' dsb ' : ' dan sebagainya ', ' good jobs ' : ' good ', ' worth ' : ' good ', 
                ' sgt ' : ' sangat ', ' terima kasih ' : ' terimakasih ', ' tengkyu ' : ' terimakasih ', 
                ' blom ' : ' belum ', ' mentok ' : ' berhenti ', ' mgkn ' : ' mungkin ', ' pegel ' : ' pegal ', 
                ' nyusahin ' : ' susah ', ' ndak ' : ' nggak ', ' tai ' : ' taik ', ' bet ' : ' banget ', ' anjeng ' : ' anjing ', 
                ' log in ' : ' masuk ', ' kdg ' : ' kadang ', ' mo ' : ' mau ', ' bs ' : ' bisa ', ' ksh ' : ' kasih ', 
                ' mon ' : ' mohon ', ' blum ' : ' belum ', ' rek ' : ' rekening ', ' lwt ' : ' lewat ', ' kemaren ' : ' kemarin ', 
                ' telp ' : ' smartphone ', ' gimna ' : ' bagaimana ', ' dkt ' : ' dekat ', ' trf ' : ' transfer ', 
                ' gmna ' : ' bagaimana ', ' kelen ' : ' kalian ', ' logout ' : ' keluar ', ' rb ' : ' ribu ', ' spt ' : ' seperti ', 
                ' aplgi ' : ' apa lagi ', ' hrs ' : ' harus ', ' verif ' : ' verifikasi ', ' cpt ' : ' cepat ', 
                ' lncr ' : ' lancar ', ' dn ' : ' dan ', ' sllu ' : ' selalu ', ' slalu ' : ' selalu ', 
                ' koplak ' : ' koplo ', ' masi ' : ' masih ', ' acc ' : ' terima ', ' gnti ' : ' ganti ', ' kok ' : ' mengapa ', 
                ' ancur ' : ' hancur ', ' system ' : ' sistem ', ' dtg ' : ' datang ', ' lsg ' : ' langsung ', 
                ' mngkn ' : ' mungkin ', ' kog ' : ' mengapa ', ' koq ' : ' mengapa ', ' klok ' : ' kalau ', 
                ' kalou ' : ' kalau ', ' sariah ' : ' syariah ', ' aktifasi ' : ' aktivasi ', ' mintak ' : ' minta ', 
                ' ortu ' : ' orang tua ', ' lgsg ' : ' langsung ', ' drpd ' : ' daripada ', ' stag ' : ' stagnan ', 
                ' cb ' : ' coba ', ' kmrin ' : ' kemarin ', ' tdak ' : ' tidak ', ' tarek ' : ' tarik ', 
                ' kya ' : ' seperti ', ' dst ' : ' dan seterusnya ', ' anjir ' : ' anjing ', ' lola ' : ' lemot ', 
                ' samsek ' : ' sama sekali ', ' ampe ' : ' sampai ', ' musti ' : ' harus ', ' msuk ' : ' masuk ', 
                ' slah ' : ' salah ', ' hr ' : ' hari ', ' bego ' : ' bodoh ', ' ad ' : ' ada ', ' spy ' : ' supaya ', 
                ' spya ' : ' supaya ', ' jmlah ' : ' jumlah ', ' tlng ' : ' tolong ', ' klw ' : ' kalau ', 
                ' gblk ' : ' goblok ', ' urgent ' : ' urgent ', ' dateng ' : ' datang ', ' bnyk ' : ' banyak ', 
                ' mna ' : ' mana ', ' temen ' : ' teman ', ' ndk ' : ' nggak ', ' kmren ' : ' kemarin ', 
                ' bbrp ' : ' beberapa ', ' cz ' : ' karena ', ' semalem ' : ' kemarin ', ' apl ' : ' aplikasi ', 
                ' kalu ' : ' kalau ', ' jmn ' : ' jaman ', ' bukak ' : ' buka ', ' doank ' : ' saja ', 
                ' bnget ' : ' banget ', ' bgus ' : ' bagus ', ' ngaco ' : ' kacau ', ' ferivikasi ' : ' verifikasi ', 
                ' anj ' : ' anjing ', ' viripikasi ' : ' verifikasi ', ' bgmn ' : ' bagaimana ', ' perifikasi ' : ' verifikasi ', 
                ' kluar ' : ' keluar ', ' sndiri ' : ' sendiri ', ' msh ' : ' masih ', ' pdhl ' : ' padahal ', 
                ' bkl ' : ' bakal ', ' erorr ' : ' eror ', ' smp ' : ' sampai ', ' dl ' : ' dulu ', ' bbrpa ' : ' beberapa ', 
                ' gax ' : ' nggak ', ' smpe ' : ' sampai ', ' mhon ' : ' mohon ', ' kga ' : ' nggak ', ' slama ' : ' selama ', 
                ' sblm ' : ' sebelum ', ' egk ' : ' nggak ', ' bego ' : ' bodoh ', ' begok ' : ' bodoh ', 
                ' bapuk ' : ' jelek ', ' stelah ' : ' setelah ', ' mao ' : ' mau ', ' smpai ' : ' sampai ', 
                ' skrang ' : ' sekarang ', ' trimakasih ' : ' terimakasih ', ' trims ' : ' terimakasih ', ' aq ' : ' aku ', 
                ' prbaiki ' : ' perbaiki ', ' sruh ' : ' suruh ', ' byr ' : ' bayar ', '  kpn ' : ' kapan ', 
                ' stack ' : ' stagnan ', ' bsa ' : ' bisa ', ' kayak ' : ' seperti ', ' aktofasi ' : ' aktivasi ', 
                ' bgd ' : ' banget ', ' aje ' : ' saja ', ' ttpi ' : ' tetapi ', ' ttap ' : ' tetap ', ' mlh ' : ' malah ', 
                ' cm ' : ' cuma ', ' masi ' : ' masih ', ' stlh ' : ' setelah ', ' ttep ' : ' tetap ', ' mlm ' : ' malam ', 
                ' bberapa ' : ' beberapa ', ' tp ' : ' tetapi ', ' haduh ' : ' aduh ', ' kntr ' : ' kantor ', 
                ' sj ' : ' saja ', ' ny ' : ' nya ', ' jls ' : ' jelas ', ' cbg ' : ' cabang ', ' rmh ' : ' rumah ', 
                ' nnti ' : ' nanti ', ' gtu ' : ' begitu ', ' bbrapa ' : ' beberapa ', ' mnt ' : ' minta ', 
                ' nggk ' : ' nggak ', ' hadeuh ' : ' aduh ', ' tlong ' : ' tolong ', ' maslah ' : ' masalah ', 
                ' ssh ' : ' susah ', ' lht ' : ' lihat ', ' mhn ' : ' mohon ', ' ad ' : ' ada ', ' aj ' : ' saja ', 
                ' ap ' : ' apa ', ' cumn ' : ' cuma ', ' apdet ' : ' update ', ' verifekasi ' : ' verifikasi ', 
                ' abdet ' : ' update ', ' mobil ' : ' mobile ', ' sudh ' : ' sudah ', ' kcil ' : ' kecil ', 
                ' spek ' : ' spesifikasi ', ' brapa ' : ' berapa ', ' sgala ' : ' segala ', ' kgk ' : ' nggak ', 
                ' hrus ' : ' harus ', ' untk ' : ' untuk ', ' kasi ' : ' kasih ', ' ggal ' : ' gagal ', ' pnya ' : ' punya ', 
                ' kmna ' : ' kemana ', ' cacad ' : ' cacat ', ' knapa ' : ' kenapa ', ' pinjem ' : ' pinjam ', 
                ' mncul ' : ' muncul ', ' pkok ' : ' pokok ', ' ktr ' : ' kantor ', ' gajelas ' : ' gaje ',  
                ' smkn ' : ' semakin ', ' kdang ' : ' kadang ', ' pngen ' : ' pengen ', ' slh ' : ' salah ', 
                ' hri ' : ' hari ', ' bodo ' : ' bodoh ', ' bnyak ' : ' banyak ', ' klu ' : ' kalau ', 
                ' laen ' :  ' lain ', ' tuk ' : ' untuk ', ' nie ' : ' ini ', ' pekok ' : ' bodoh ', 
                ' sebelom ' : ' sebelum ', ' galat ' : ' gagal ', ' nnya ' : ' nanya ', ' hpus ' : ' hapus ', 
                ' ato ' : ' atau ', ' kureng ' : ' kurang ', ' stiap ' : ' setiap ', ' sprti ' : ' seperti ', 
                ' sblh ' : ' sebelah ', }
        
        def normalisasi(str_text):
            for i in norm:
                str_text = str_text.replace(i, norm[i])
            return str_text
        
        df['content'] = df['content'].apply(lambda x: normalisasi(x))
        df['content'] = df['content'].str.replace(r'\bnya\b', '', regex=True).str.strip()
        st.dataframe(df.head())

        st.header("Labelling")
        # # baca kamus indonesia sentiment lexicon (positif dan negatif)
        # positive_lexicon = set(pd.read_csv("kamus_indonesia_sentiment_lexicon/positive.tsv", sep="\t", header=None)[0])
        # negative_lexicon = set(pd.read_csv("kamus_indonesia_sentiment_lexicon/negative.tsv", sep="\t", header=None)[0])

        # # # fungsi untuk menentukan sentimen
        # def determine_sentiment(text):
        #     positive_count = sum(1 for word in text.split() if word in positive_lexicon)
        #     negative_count = sum(1 for word in text.split() if word in negative_lexicon)
        #     if positive_count > negative_count:
        #         return "POSITIVE"
        #     elif positive_count < negative_count:
        #         return "NEGATIVE"
        #     else:
        #         return "NEUTRAL"

        # # tentukan sentimen untuk setiap ulasan 
        # df['sentiment'] = df['content'].apply(determine_sentiment)
        # st.dataframe(df.head())

        # LABELLING
        # load lexicon positive and negative data
        lexicon_positive = dict()
        import csv
        with open("/aplikasi_skripsi/kamus_indonesia_sentiment_lexicon/new_positive.csv", "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # lexicon_positive[row[0]] = int(row[1])
                if len(row) >= 2:  # pastikan setiap baris memiliki minimal dua kolom
                    lexicon_positive[row[0]] = int(row[1])

            lexicon_negative = dict()
            import csv
            with open("/aplikasi_skripsi/kamus_indonesia_sentiment_lexicon/new_negative.csv", "r") as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    # lexicon_negative[row[0]] = int(row[1])
                    if len(row) >= 2:  # pastikan setiap baris memiliki minimal dua kolom
                        lexicon_negative[row[0]] = int(row[1])

            # function to determine sentiment polarity
            def sentiment_analysis_lexicon_indonesia(text):
                # for word in text
                score = 0

                # Split text menjadi list of words (gunakan space sebagai pemisah)
                words = text.split()

                for word in words:
                    if (word in lexicon_positive):
                        score = score + lexicon_positive[word]
                for word in words:
                    if (word in lexicon_negative):
                        score = score + lexicon_negative[word]
                
                polarity = ''
                if (score > 0):
                    polarity = 'POSITIVE'
                elif (score < 0):
                    polarity = 'NEGATIVE'
                else:
                    polarity = 'NEUTRAL'
                return score, polarity
            
            results = df['content'].apply(sentiment_analysis_lexicon_indonesia)
            results = list(zip(*results))
            df['polarity_score'] = results[0]
            df['sentiment'] = results[1]

            st.write(df['sentiment'].value_counts())
            st.dataframe(df.head())
        # END LABELLING

        st.subheader("Menghapus ulasan dengan sentimen neutral")
        # menghapus row dengan sentimen netral
        neutral_row = df[(df.sentiment == 'NEUTRAL')].index
        df = df.drop(neutral_row)
        st.dataframe(df.head())
        st.write(f"Data has: {df.shape[0]} rows and {df.shape[1]} columns")

        st.subheader("Stopwords")
        # stopwords
        import Sastrawi
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

        more_stop_words = []

        stop_words = StopWordRemoverFactory().get_stop_words()
        stop_words.extend(more_stop_words)

        new_array = ArrayDictionary(stop_words)
        stop_words_remover_new = StopWordRemover(new_array)

        def stopword(str_text):
            str_text = stop_words_remover_new.remove(str_text)
            return str_text

        df['content_stopwords'] = df['content'].apply(lambda x: stopword(x))
        st.dataframe(df.head())

        # STEMMING DAN TOKENIZING
        st.subheader("Tokenizing dan Stemming")
        # # Initialize stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        # # Define Indonesian stopwords
        stop_words = set(stopwords.words('indonesian'))

        # # Function for preprocessing: tokenizing and stemming
        def preprocess_review(review):
            # Tokenize the review
            tokens = word_tokenize(review)
            
        #     # Remove stopwords
            tokens = [word for word in tokens if word.lower() not in stop_words]
            
        #     # Stem the tokens
            stems = [stemmer.stem(token) for token in tokens]
            
            return stems

        # # Apply the function to the content column
        df['tokenized_stemmed'] = df['content_stopwords'].apply(preprocess_review)

        # # Function to clean the data
        def clean_data(tokenized_list):
            # Convert list to a string, remove commas and brackets, and then convert back to a list of words
            clean_str = ' '.join(tokenized_list)
            return clean_str

        # # Apply the function to the 'tokenized_stemmed' column
        df['cleaned_tokenized_stemmed'] = df['tokenized_stemmed'].apply(clean_data)
        # END STEMMING DAN TOKENIZING

        # Untuk Development, membaca dataset yang telah di preprocessing (dikarenakan proses stemming lama)
        # df = pd.read_csv("preprocessed/preprocessed_data_bsi.csv")

        st.dataframe(df.head())

        # kode untuk mengeksport ke csv
        df.to_csv('preprocessed/preprocessed_data_bsi.csv', index=False)

        st.subheader("Jumlah ulasan dengan sentimen positif dan negatif")
        # jumlah ulasan dengan sentimen positif dan negatif
        sentiment_counts = df.sentiment.value_counts()
        st.write(sentiment_counts)

        # visualisasi
        st.header("Visualization")

        # BARCHART
        # Tentukan palet warna secara manual
        palette = {'POSITIVE': 'green', 'NEGATIVE': 'maroon'}
        # Buat figure untuk plot
        fig, ax = plt.subplots(figsize=(4, 4))
        # Buat countplot dengan Seaborn, menggunakan palet warna yang telah ditentukan
        sns.countplot(x='sentiment', data=df, palette=palette, ax=ax)
        # Atur judul dan label sumbu
        ax.set_title('Distribusi Sentimen')
        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Jumlah')
        # Tampilkan plot di Streamlit
        st.pyplot(fig)
        # END BARCHART

        # PIECHART
        # Tentukan palet warna secara manual
        palette = {'POSITIVE': 'lightgreen', 'NEGATIVE': 'coral'}
        # Hitung jumlah setiap kategori sentimen
        sentiment_counts = df['sentiment'].value_counts()
        # Buat figure untuk pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        # Buat pie chart
        ax.pie(sentiment_counts, labels=sentiment_counts.index, colors=[palette[key] for key in sentiment_counts.index], autopct='%1.1f%%', startangle=90)
        # Atur judul
        ax.set_title('Distribusi Sentimen')
        # Tampilkan plot di Streamlit
        st.pyplot(fig)
        # END PIECHART

        # WORDCLOUD
        st.subheader("Wordcloud")
        # Filter data berdasarkan sentimen
        data_negatif = df[df['sentiment'] == 'NEGATIVE']
        data_positif = df[df['sentiment'] == 'POSITIVE']

        # Gabungkan semua kata positif menjadi satu string, konversikan setiap elemen ke string
        all_text_s1 = ' '.join(str(word) for word in data_positif['cleaned_tokenized_stemmed'])

        # Buat WordCloud untuk kata-kata positif
        wordcloud_positif = WordCloud(colormap='Greens', width=1000, height=1000, mode='RGBA',
                                    background_color='white').generate(all_text_s1)

        # Plot WordCloud untuk kata-kata positif menggunakan Matplotlib
        fig_positif, ax_positif = plt.subplots(figsize=(6, 6))
        ax_positif.imshow(wordcloud_positif, interpolation='bilinear')
        ax_positif.axis('off')
        ax_positif.set_title('Visualisasi Kata Positif', color='black')
        plt.margins(x=0, y=0)
        # Tampilkan plot positif di Streamlit
        st.pyplot(fig_positif)

        # Gabungkan semua kata negatif menjadi satu string, konversikan setiap elemen ke string
        all_text_s0 = ' '.join(str(word) for word in data_negatif['cleaned_tokenized_stemmed'])
        # Buat WordCloud untuk kata-kata negatif
        wordcloud_negatif = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA',
                                    background_color='white').generate(all_text_s0)
        # Plot WordCloud untuk kata-kata negatif menggunakan Matplotlib
        fig_negatif, ax_negatif = plt.subplots(figsize=(6, 6))
        ax_negatif.imshow(wordcloud_negatif, interpolation='bilinear')
        ax_negatif.axis('off')
        ax_negatif.set_title('Visualisasi Kata Negatif', color='black')
        plt.margins(x=0, y=0)
        # Tampilkan plot negatif di Streamlit
        st.pyplot(fig_negatif)

        # FREKUENSI KATA
        # Menggabungkan semua token dari setiap baris dalam kolom 'tokenized_stemmed'
        all_tokens = []

        # Mengubah setiap ulasan di kolom 'ulasan' menjadi list kata
        data_ulasan = df['cleaned_tokenized_stemmed'].apply(lambda x: x.split() if isinstance(x, str) else []) 

        for sublist in data_ulasan:
            all_tokens.extend(sublist)

        # Menghitung distribusi frekuensi
        freq_dist = FreqDist(all_tokens)
        # Menampilkan frekuensi kata dalam Streamlit
        st.subheader("Frekuensi Kata")
        # st.write("Berikut adalah jumlah frekuensi setiap kata:")
        # Convert frequency distribution to a dictionary and display it
        freq_dict = dict(freq_dist)
        # st.write(freq_dict)
        # Atau untuk menampilkan top N kata paling umum
        st.write("Top 10 Kata Paling Umum:")
        st.dataframe(freq_dist.most_common(10))

        # Membuat plot frekuensi
        st.subheader("Plot Frekuensi Kata")

        # Membuat figure dan axis
        fig, ax = plt.subplots()
        # Mengambil 30 kata paling umum
        most_common = freq_dist.most_common(10)
        # Memisahkan kata dan frekuensinya
        words, frequencies = zip(*most_common)

        # Membuat bar plot
        ax.bar(words, frequencies)
        ax.set_xticklabels(words, rotation=90)
        ax.set_title("Top 10 Kata Paling Umum")
        ax.set_xlabel("Kata")
        ax.set_ylabel("Frekuensi")

        # Menampilkan plot dalam Streamlit
        st.pyplot(fig)
        # END FREKUENSI KATA

        # SPLITTING DATA
        st.header("Splitting Data")
        df = df.dropna()
        X = df.cleaned_tokenized_stemmed
        y = df.sentiment
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.subheader("Data Train")
        st.dataframe(x_train)
        st.write(f"Data Train sebanyak : {x_train.count()}")
        st.subheader("Data Test")
        st.dataframe(x_test)
        st.write(f"Data Test sebanyak : {x_test.count()}")

        # TRAINING DATA
        st.header("Traning Data dengan Algoritma Multinomial Naive Bayes")

        tvec = TfidfVectorizer()
        clf = MultinomialNB()

        model = Pipeline([('vectorizer', tvec), ('classifier', clf)])
        model.fit(x_train, y_train)

        st.write(f"Akurasi model setelah proses training : {model.score(x_test, y_test)}")

        hasil = model.predict(x_test)
        matrix = classification_report(y_test, hasil)
        st.write(f"Classtification Report : \n {matrix}")

        # HYPERPARAMETER TUNING
        st.header("Hyperparameter Tuning Algoritma Multinomial Naive Bayes")

        # Define a pipeline combining a CountVectorizer and a MultinomialNB classifier
        pipeline = Pipeline([
            ('vect', TfidfVectorizer()),  # Convert text data to a matrix of token counts
            ('nb', MultinomialNB())       # Apply Multinomial Naive Bayes classifier
        ])

        # Define the parameter grid to search
        param_grid = {
            'vect__ngram_range': [(1, 1), (1, 2)],  # Unigrams or unigrams + bigrams
            'nb__alpha': [0.1, 0.5, 1.0],           # Different values for Laplace smoothing
            'nb__fit_prior': [True, False],         # Whether to learn class prior probabilities
        }

        # Perform GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)

        st.dataframe(grid_search.cv_results_)

        # # Print the best parameters and the best score
        st.write("Best Parameters:", grid_search.best_params_)
        st.write("Best Cross-Validation Score:", grid_search.best_score_)

        # SIMPAN MODEL KE FILE PIKEL
        joblib.dump(grid_search, 'model/model_bsi.pkl')
        # END SIMPAN MODEL KE FILE PIKEL

        # EVALUASI DENGAN CONFUSSION MATRIX
        st.header("Evaluasi dengan Confussion Matrix")
        # # Evaluate the model on the test data
        akurasi_model_train = grid_search.score(x_train, y_train)
        # akurasi_model_test = grid_search.score(x_test, y_test)
        st.write(f"Akurasi model dengan data latih : {akurasi_model_train}")
        # st.write(f"Akurasi model dengan data test : {akurasi_model_test}")

        y_pred = grid_search.predict(x_test)
        st.write(f"Classtification Report : \n {classification_report(y_test, y_pred)}")

        #confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # st.write(f"Confusion Matrix")
        st.write(f"True Positive : {tp}")
        st.write(f"False Positive : {fp}")
        st.write(f"True Negative : {tn}")
        st.write(f"False Negative : {fn}")
        # END EVALUASI DENGAN CONFUSSION MATRIX

        # HEATMAP CONFUSION MATRIX
        # Confusion matrix calculation (using the same y_test and y_pred as in your code)
        cm = confusion_matrix(y_test, y_pred)
        # Create a heatmap using seaborn
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and an axis object
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, 
                    xticklabels=['Predicted Negative', 'Predicted Positive'], 
                    yticklabels=['Actual Negative', 'Actual Positive'], ax=ax)

        # Set plot labels and title
        ax.set_title('Confusion Matrix Heatmap')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        # Display the heatmap in Streamlit
        st.pyplot(fig)
        # END HEATMAP CONFUSION MATRIX

        # VIUALISASI DAN FREKUENSI DATA TEST
        st.subheader("Visualisasi Data Test")
        data_test = pd.DataFrame(x_test)
        data_test['sentiment'] = y_test
        # st.dataframe(data_test)

        # PIECHART DATA TEST
        # Tentukan palet warna secara manual
        palette = {'POSITIVE': 'lightgreen', 'NEGATIVE': 'coral'}
        # Hitung jumlah setiap kategori sentimen
        sentiment_counts = data_test['sentiment'].value_counts()
        # Buat figure untuk pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        # Buat pie chart
        ax.pie(sentiment_counts, labels=sentiment_counts.index, colors=[palette[key] for key in sentiment_counts.index], autopct='%1.1f%%', startangle=90)
        # Atur judul
        ax.set_title('Distribusi Sentimen Data Test')
        # Tampilkan plot di Streamlit
        st.pyplot(fig)
        # END PIECHART DATA TEST

        # PISAHKAN DATA TEST (POSITIVE & NEGATIVE)
        data_test_negative = data_test[data_test['sentiment'] == 'NEGATIVE']
        data_test_positive = data_test[data_test['sentiment'] == 'POSITIVE']

        # END PISAHKAN DATA TEST (POSITIVE & NEGATIVE)

        # FREKUENSI KATA POSITIVE
        # Menggabungkan semua token dari setiap baris dalam kolom 'tokenized_stemmed'
        all_positive_test_tokens = []
        # data_ulasan = df['cleaned_tokenized_stemmed']

        # Mengubah setiap ulasan di kolom 'ulasan' menjadi list kata
        data_ulasan_positive_test = data_test_positive['cleaned_tokenized_stemmed'].apply(lambda x: x.split())

        for sublist in data_ulasan_positive_test:
            all_positive_test_tokens.extend(sublist)

        # Menghitung distribusi frekuensi
        freq_dist_positive = FreqDist(all_positive_test_tokens)
        # Menampilkan frekuensi kata dalam Streamlit
        st.subheader("Frekuensi Kata Positive Data Test")
        # st.write("Berikut adalah jumlah frekuensi setiap kata:")
        # Convert frequency distribution to a dictionary and display it
        freq_dict = dict(freq_dist_positive)
        # st.write(freq_dict)
        # Atau untuk menampilkan top N kata paling umum
        st.write("Top 10 Kata Paling Umum:")
        st.dataframe(freq_dist_positive.most_common(10))
        # END FREKUENSI KATA POSITIVE

        # FREKUENSI KATA NEGATIVE
        # Menggabungkan semua token dari setiap baris dalam kolom 'tokenized_stemmed'
        all_negative_test_tokens = []
        # data_ulasan = df['cleaned_tokenized_stemmed']

        # Mengubah setiap ulasan di kolom 'ulasan' menjadi list kata
        data_ulasan_negative_test = data_test_negative['cleaned_tokenized_stemmed'].apply(lambda x: x.split())

        for sublist in data_ulasan_negative_test:
            all_negative_test_tokens.extend(sublist)

        # Menghitung distribusi frekuensi
        freq_dist_negative = FreqDist(all_negative_test_tokens)
        # Menampilkan frekuensi kata dalam Streamlit
        st.subheader("Frekuensi Kata Negative Data Test")
        # st.write("Berikut adalah jumlah frekuensi setiap kata:")
        # Convert frequency distribution to a dictionary and display it
        freq_dict = dict(freq_dist_negative)
        # st.write(freq_dict)
        # Atau untuk menampilkan top N kata paling umum
        st.write("Top 10 Kata Paling Umum:")
        st.dataframe(freq_dist_negative.most_common(10))
        # END FREKUENSI KATA NEGATIVE
        # END VIUALISASI DAN FREKUENSI DATA TEST

    else:
        st.write("Silakan unggah file CSV untuk melihat data.")


if __name__ == "__main__":
    main()

# Footer
st.markdown("""
---
Made with ❤️ in Lhokseumawe by *Mr Mustache* 🥸. All rights reserved.
""")
