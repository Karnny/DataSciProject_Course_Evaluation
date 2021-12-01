

import streamlit as st
import pandas as pd
import pickle as pk
import bz2

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pk.load(data)
    return data


# !Header App
st.write("""# Course enrollment evaluation 
Topic description:
Evaluate how students of each school perform in each course for adjusting the suitable course 
for the next academic year.

**Group member:** \n
1. 6231302021 Thanchanok  Thiju \n
2. 6231302017 Sahapol Polyiam \n
3. 6231302015 Siwakiat Karnpattaranont \n
4. 6231302012 Phornwanat Boonman \n

""")


student_grade = pd.read_csv('student_grade.csv')
course_data = student_grade.COURSENAMEENG.unique()
school_data = student_grade.SCHOOL.unique()
subgroup_data = student_grade.SUBGROUP.unique()
maingroup_data = student_grade.MAINGROUP.unique()
sample_file = None
# !Navbar
st.sidebar.header('Select data')
course = st.sidebar.selectbox('Choose Course Name', course_data)
school = st.sidebar.selectbox('Choose School', school_data)
maingroup = st.sidebar.selectbox('Choose Main Group', maingroup_data)

st.sidebar.header('Or upload a file')

data = {
    'COURSENAMEENG': course,
    'SCHOOL': school,
    'MAINGROUP': maingroup
}


sample_file = st.sidebar.file_uploader('', type=['.csv'], accept_multiple_files=False,
                                       key=None, help=None, on_change=None, args=None, kwargs=None)

def handle_file_upload(sample):
    st.subheader('User input: Uploaded')
    df = pd.read_csv(sample)
    st.write(df)
    do_predict(df)

def do_predict(df):
    
    # ?data is display by user input
    st.subheader('Pre-Process inputs')
    courseData = pd.get_dummies(df[['COURSENAMEENG']])
    st.write("""###### Course""")
    st.write(courseData)

    # ?data is display by user input
    schoolData = pd.get_dummies(df[['SCHOOL']])
    st.write("""###### School""")
    st.write(schoolData)

    # ?data is display by user input
    maingroupData = pd.get_dummies(df[['MAINGROUP']])
    st.write("""###### Main Group""")
    st.write(maingroupData)

    all_input_features = pd.concat([courseData, schoolData, maingroupData], axis=1)
    train_features = pd.read_csv('all_features.csv')

    # Combine input-features with trained-features
    all_features = train_features.append(all_input_features, sort=False)
    # Fill the rest N/A, NaN, features with 0 values
    all_features = all_features.fillna(0)


    st.subheader('All Features')
    st.write(all_features)

    st.subheader('Normalization')
    normalizer = pk.load(open('normalization.pkl', 'rb'))
    # knn = pk.load(open('best_knn.pkl', 'rb'))
    knn = decompress_pickle('best_knn_compressed.pbz2')

    X_new = normalizer.transform(all_features)
    st.write(X_new)


    st.subheader('Prediction')
    prediction = knn.predict(X_new)
    st.write(prediction)

if sample_file is not None:
    handle_file_upload(sample_file)

else:
    # !body App
    st.subheader('User input')
    df = pd.DataFrame(data, index=[0])
    st.write(df)
    do_predict(df)
    


