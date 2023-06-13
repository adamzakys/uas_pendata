import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


import pickle

from sklearn import metrics

st.set_page_config(
    page_title="Prediksi Kemungkinan pendonor akan mendonorkan"
)

st.title('Prediksi Kemungkinan pendonor akan mendonorkan')
st.write("""
Aplikasi Untuk Memprediksi Kemungkinan pendonor akan mendonorkan darahnya pada saat kendaraan datang ke kampus berikutnya.
""")
st.write("""
Nama : Muhammad Adam Zaky Jiddyansah
""")
st.write("""
NIM : 210411100234
""")

tab1, tab2, tab3, tab4 = st.tabs(["Data Understanding", "Preprocessing", "Modelling", "Implementation"])

with tab1:
    st.write("""
    <h5>Data Understanding</h5>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Dataset:
    <a href="https://www.kaggle.com/datasets/shabbir94/blood-transfusion"> https://www.kaggle.com/datasets/shabbir94/blood-transfusion</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Repository Github
    https://raw.githubusercontent.com/adamzakys/SourceFiles/main/PenambanganData/transfusion.csv
    """, unsafe_allow_html=True)
    
    st.write('Type dataset ini adalah Real')
    st.write('Dataset ini berisi tentang klasifikasi apakah pendonor akan mendonorkan darahnya pada saat kendaraan datang ke kampus berikutnya.')
    df = pd.read_csv("https://raw.githubusercontent.com/adamzakys/SourceFiles/main/PenambanganData/transfusion.csv")
    st.write("Dataset Blood Transfusion : ")
    st.write(df)
    st.write("Penjelasan kolom-kolom yang ada")

    st.write("""
    <ol>
    <li>Recency : bulan sejak donasi terakhir</li>
    <li>Frekuensi : Jumlah Donasi</li>
    <li>Monetary : Total darah yang didonorkan dalam cc</li>
    <li>Time : bulan sejak donasi pertama</li>
    <li>whether he/she donated blood in March 2007 : variabel biner yang mewakili apakah dia mendonor darah pada Maret 2007 (1 berarti mendonor darah; 0 berarti tidak mendonor darah).</li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("""
    <h5>Preprocessing Data</h5>
    <br>
    """, unsafe_allow_html=True)
    st.write("""
    <p style="text-align: justify;text-indent: 45px;">Preprocessing data adalah proses mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini diperlukan untuk memperbaiki kesalahan pada data mentah yang seringkali tidak lengkap dan memiliki format yang tidak teratur. Preprocessing melibatkan proses validasi dan imputasi data.</p>
    <p style="text-align: justify;text-indent: 45px;">Salah satu tahap Preprocessing data adalah Normalisasi. Normalisasi data adalah elemen dasar data mining untuk memastikan record pada dataset tetap konsisten. Dalam proses normalisasi diperlukan transformasi data atau mengubah data asli menjadi format yang memungkinkan pemrosesan data yang efisien.</p>
    <br>
    """,unsafe_allow_html=True)
    scaler = st.radio(
    "Pilih metode normalisasi data",
    ('Tanpa Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        scaler = MinMaxScaler()
        df_for_scaler = pd.DataFrame(df, columns = ['Recency','Frequency','Monetary','Time'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['Recency','Frequency','Monetary','Time'])
        df_drop_column_for_minmaxscaler=df.drop(['Recency','Frequency','Monetary','Time'], axis=1)
        df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    st.write(df_new)

with tab3:
    st.write("""
    <h5>Modelling</h5>
    <br>
    """, unsafe_allow_html=True)

    nb = st.checkbox("Naive Bayes")  # Checkbox for Naive Bayes
    knn = st.checkbox("KNN")  # Checkbox for KNN
    ds = st.checkbox("Decision Tree")  # Checkbox for Decision Tree
    mlp = st.checkbox("MLP")  # Checkbox for MLP

    # Splitting the data into features and target variable
    X = df.drop('whether he/she donated blood in March 2007', axis=1)
    y = df['whether he/she donated blood in March 2007']

    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = []  # List to store selected models

    if nb:
        models.append(('Naive Bayes', GaussianNB()))
    if knn:
        models.append(('KNN', KNeighborsClassifier()))
    if ds:
        models.append(('Decision Tree', DecisionTreeClassifier()))
    if mlp:
        models.append(('MLP', MLPClassifier()))

    if len(models) == 0:
        st.warning("Please select at least one model.")

    else:
        accuracy_scores = []  # List to store accuracy scores

        st.write("<h6>Accuracy Scores:</h6>", unsafe_allow_html=True)
        st.write("<table><tr><th>Model</th><th>Accuracy</th></tr>", unsafe_allow_html=True)

        for model_name, model in models:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            st.write("<tr><td>{}</td><td>{:.2f}</td></tr>".format(model_name, accuracy), unsafe_allow_html=True)

        st.write("</table>", unsafe_allow_html=True)

        # Displaying the table of test labels and predicted labels
        st.write("<h6>Test Labels and Predicted Labels:</h6>", unsafe_allow_html=True)
        labels_df = pd.DataFrame({'Test Labels': y_test, 'Predicted Labels': y_pred})
        st.write(labels_df)


# Define the decision tree classifier model
model = DecisionTreeClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Save the decision tree model as a pickle file
filename = 'decision_tree.pkl'
pickle.dump(model, open(filename, 'wb'))


with tab4:
    st.write("""
    <h5>Implementation</h5>
    <br>
    """, unsafe_allow_html=True)
    
    # Load the decision tree model from the pickle file
    filename = 'decision_tree.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))

    # Define the input values for prediction
    recency = st.number_input("Recency (bulan sejak donasi terakhir):")
    frequency = st.number_input("Frequency (jumlah donasi):")
    monetary = st.number_input("Monetary (total darah yang didonorkan dalam cc):")
    time = st.number_input("Time (bulan sejak donasi pertama):")

    # Create a dataframe with the input values
    input_data = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary], 'Time': [time]})

    # Preprocess the input data
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    prediction = loaded_model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.write("Pendonor kemungkinan akan mendonorkan darah pada saat kendaraan datang ke kampus berikutnya.")
    else:
        st.write("Pendonor kemungkinan tidak akan mendonorkan darah pada saat kendaraan datang ke kampus berikutnya.")