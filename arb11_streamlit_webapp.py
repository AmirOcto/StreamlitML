# Import libraries
from itertools import count
from unicodedata import category
from charset_normalizer import detect
import streamlit as st
import hydralit_components as hc
import datetime

import pandas as pd
import numpy as np

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import io

import plotly
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px

from pathlib import Path
import base64

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from fancyimpute import KNN

# Initial page config

st.set_page_config(
     page_title='Marketing App',
     layout="wide",
     page_icon= ':computer:',
     initial_sidebar_state="expanded"
)

def img_to_bytes(img_path):
    #to encode an image into a string
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=150 height=85>](https://www.aub.edu.lb/osb/MSBA/Pages/default.aspx)'''.format(img_to_bytes("AUB.png")), 
                    unsafe_allow_html=True)

#-------------------------------------
#--------------SIDEBAR----------------
#-------------------------------------

st.sidebar.header('All in One App')

with st.sidebar:
        company_name = st.text_input("Enter The Company Name")

with st.sidebar: 
        fileupload = st.file_uploader("Upload a dataset", type = ("csv", "xlsx"))
        if fileupload is None:
                st.write("waiting...")
        else:
                data = pd.read_csv(fileupload)
        

with st.sidebar: 
        fileuploadclean = st.file_uploader("Upload the cleaned file", type = ("csv", "xlsx"))
        if fileuploadclean is None:
                st.write("Upload a file")
        else:
                dataclean = pd.read_csv(fileuploadclean)
                st.write(f'The dataset {fileuploadclean.name} contains {dataclean.shape[0]} rows and {dataclean.shape[1]} columns.')  

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('Amir Bazzi | Msba 370 2022 \n\n arb11@mail.aub.edu ')

st.title("The Business and Marketing App")
st.subheader("Prepare. Explore. Analyze.")

#-------------------------------------
#--------------NAV BAR----------------
#-------------------------------------

# specify the primary menu definition
menu_data = [
    
    {'id':'data_preparation',
    'icon':"",
    'label':"Data Preparation",
    },

    {'id':'data_inspection',
    'icon': "", 
    'label':"Data Inspection",
    'submenu':[{'id':'subid11','icon': "", 'label':"General Profiling"},
               {'id':'subid12','icon': "", 'label':"Detailed Profiling"}]},


#     {'id':'data_analysis',
#     'icon': "",
#     'label':"Data Analysis", 
#     'submenu':[{'id':'subid21','icon': "", 'label':"Univariate Analysis"},
#                {'id':'subid22','icon': "", 'label':"Bivariate Analysis"}]},


     {'id':'crm',
        'icon': "", 
        'label':"Customer Analysis",
        'submenu':[{'id':'subid31','icon': "", 'label':"RFM Analysis Clustering"},
                {'id':'subid32','icon': "", 'label':"Reviews NLP"}]},

    {'id':'business_analytics',
     'icon': "far fa-chart-bar", 
     'label':"Business Analytics"},

    {'id':' arm',
    'icon': "", 
    'label':"Market Basket Analysis"},

    {'id':'recommender_system',
    'icon': "fa-solid fa-radar",
    'label':"Recommender System" },
]

over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    login_name='Contact Info',
    hide_streamlit_markers=False, 
    sticky_nav=True, #at the top or not
    sticky_mode = 'pinned', #jumpy or not-jumpy, but sticky or pinned
)
      

def missing_zero_values_table(df):
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        return mz_table

#---------------------------------------------
#--------------Data Inspection----------------
#---------------------------------------------

if menu_id == "subid11":
# Upload dataset
        #General Inspection
        #Preview dataset
        st.header("General Inspection")
        if fileuploadclean is not None:
                st.write(f'The dataset {fileuploadclean.name} contains {dataclean.shape[0]} rows and {dataclean.shape[1]} columns.')

                if st.checkbox("Preview Dataset"):
                        if st.button("Head"):
                                st.write(dataclean.head())
                        elif st.button("Tail"):
                                st.write(dataclean.tail())
                        else:
                                rows = st.slider("Select Number of Rows", 2, dataclean.shape[0])
                                st.write(dataclean.head(rows))
                
                #Column names, dimension, info, datatypes and summary
                if st.checkbox("Show Column Names"):
                        st.write(dataclean.columns)
                if st.checkbox("Show Dimensions"):
                        st.write(dataclean.shape)
        

if menu_id == "subid12":
        st.header("Detailed Inspection")
        if fileuploadclean is not None:
                # Detailed Inspection
                profile = ProfileReport(dataclean,
                        dataset={
                        "description": f'This profiling report was generated for {company_name}',
                        "url": f"https://www.{company_name}.com/blog/",
                })
                st_profile_report(profile)


#---------------------------------------------
#--------------Data Preparation---------------
#---------------------------------------------

if menu_id == "data_preparation":
        st.header("Data Preparation")
        if fileupload is not None:
                st.write(f'The dataset {fileupload.name} contains {data.shape[0]} rows and {data.shape[1]} columns.')

                #Data Types handling
        
        st.subheader("Data Types")
        if fileupload is not None: 
                if st.checkbox("Show Data Types"):
                        st.write(data.dtypes.astype(str))

                cat_cols = st.multiselect('Choose columns to be converted into float numerical columns', 
                options=[c for c in data.columns.values], 
                )

                if st.checkbox("Convert to float"):
                        data[cat_cols] = data[cat_cols].astype(np.float64)
                        st.write(data[cat_cols].dtypes.astype(str))

                num_cols = st.multiselect('Choose columns to be converted into categorical columns', 
                options=[c for c in data.select_dtypes(include=[int, float]).columns.values], 
                )

                if st.checkbox("Convert to category"):
                        data[num_cols]= data[num_cols].astype('category')
                        st.write(data[num_cols].dtypes.astype(str))


                #Data Cleaning

                st.subheader("Missing Values")
                
                if st.checkbox("Show Missing Values"):
                        st.dataframe(missing_zero_values_table(data).astype(str))
                
                col1, col2 =st.columns(2)
                with col1:
                          st.subheader("Numerical Variables Missing Values")
                          numeric_cols = st.multiselect("Select numeric columns to treat", options=[c for c in data.select_dtypes(include=[int, float]).columns.values] )
                          #Methods to treat missing values
                          if len(numeric_cols)>0:
                                missing_value_method = st.selectbox("Select Missing Values Handling Technique", ("Remove","Replace with Median", "Replace with Mean","Replace with Mode"))
                                if missing_value_method == "Replace with Median":
                                        for column in data[numeric_cols].columns:
                                                data[column].fillna(data[column].median(), inplace=True)
                                        #Check if the missing values were removed
                                                #if st.checkbox("Show Numerical Missing Values After Treatment"):
                                        st.write(f'Results after treating {numeric_cols} using {missing_value_method}')
                                        st.write(data[numeric_cols].isnull().sum())

                                elif missing_value_method == "Replace with Mean":
                                        for column in data[numeric_cols].columns:
                                                data[column].fillna(data[column].mean(), inplace=True)
                                        #Check if the missing values were removed
                                        st.write(f'Results after treating {numeric_cols} using {missing_value_method}')
                                        st.write(data[numeric_cols].isnull().sum())

                                elif missing_value_method == "Replace with Mode":
                                        for column in data[numeric_cols].columns:
                                                data[column].fillna(data[column].mode()[0], inplace=True)
                                        #Check if the missing values were removed
                                        st.write(f'Results after treating {numeric_cols} using {missing_value_method}')
                                        st.write(data[numeric_cols].isnull().sum())
                                
                                else :
                                        for column in data[numeric_cols].columns:
                                                data= data.dropna( how='any',subset=[column])
                                        #Check if the missing values were removed
                                        st.write(f'Results after treating {numeric_cols} using {missing_value_method}')
                                        st.write(data[numeric_cols].isnull().sum())
             
                with col2:
                        st.subheader("Categorical Variables Missing Values")
                        cat_cols = st.multiselect("Select categorical columns to treat", options=[c for c in data.select_dtypes(exclude=[int, float]).columns.values] )
                        #Methods to treat missing value
                        if len(cat_cols)>0:
                                missing_value_method = st.selectbox("Select Missing Values Handling Technique", ("Replace with Mode", "KNN Imputer Coming Soon.."))
                                if missing_value_method == "Replace with Mode":
                                        for column in data[cat_cols].columns:
                                               data[column].fillna(data[column].mode()[0], inplace=True)
                                        #Check if the missing values were removed
                                        st.write(f'Results after treating {cat_cols} using {missing_value_method}')
                                        st.write(data[cat_cols].isnull().sum())

                #Duplicates 

                st.subheader("Duplicates")
                st.write(f'There are {data.duplicated().sum()} duplicates in the dataset.')
                if st.checkbox("Remove Duplicates"):
                        data = data.drop_duplicates()
                        st.write(f'{data.duplicated().sum()} duplicates are left')
                
                st.header("Outliers")
                def remove_outlier(df_in, col_name):
                        q1 = df_in[col_name].quantile(0.25)
                        q3 = df_in[col_name].quantile(0.75)
                        iqr = q3-q1 #Interquartile range
                        fence_low  = q1-1.5*iqr
                        fence_high = q3+1.5*iqr
                        df_in = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
                        return df_in
                num_cols = st.selectbox('Choose a numerical column to remove outliers', 
                options=[c for c in data.select_dtypes(include=[int, float]).columns.values], 
                )
                if st.checkbox("Remove Outliers"):
                        remove_outlier(data, num_cols)
                
                #Download Cleaned dataset

                st.write(f'The dataset {fileupload.name} contains {data.shape[0]} rows and {data.shape[1]} columns.')

                @st.cache
                def convert_df(df):
                        return df.to_csv().encode('utf-8')


                csv = convert_df(data)

                st.download_button(
                        "Press to Download The Cleaned Dataset",
                        csv,
                        f"{company_name}_dfupdated.csv",
                        "text/csv",
                        key='download-csv'
                        )






#------------------------------------------
#------------K Means Clustering------------
#------------------------------------------
import pandas as pd 
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py

#Step0: define all functions for the sake of clarity 
#Step1: User inputs the numerical features for KMeans to cluster
#Step1.1: (Backend)- dropna 
#Step2: Take subset of the dataframe based on the chosen columns
#Step3: Scale this dataframe
#Step4: Split streamlit page into two columns to optimize for K
#Step4i: col1 will contain the k-elbow graph where the user inputs the k range
#Step4ii: col2 will contain the silhoeutte analysis where the users input the k range
#Step5: After optimizing k, cluster the data 
#Step6: Show a dataframe of each labeled cluster with the aggregated mean for monetary, frequncy, recency
#Step7: A bar chart showing the count of each cluster 
#Step8: 3d Visual of the cluster
#Step9: Download the new dataframe with the given labels 


#Step0 
def preprocess(df):
    """Preprocessing function that takes the numerical dataframe as input and prepare it for KMeans clustering"""
    df_log = np.log1p(df)
    scaler = StandardScaler()
    scaler.fit(df_log)
    normalized_rfm_split = scaler.transform(df_log)
    
    return normalized_rfm_split

def elbow_plot2(df):
    """Create elbow plot from normalized data"""
    
    normalized_rfm_split = preprocess(df)
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'random')
        kmeans.fit(df)
        #print (i,kmeans.inertia_)
        wcss.append(kmeans.inertia_) 
        #fig = plt.figure(figsize = (10, 5))
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WSS') #within cluster sum of squares
        plt.show()
        #st.pyplot(fig)

def find_k(df, increment=0, decrement=0):
    """Find the optimum k clusters"""
    
    normalized_rfm_split = preprocess(df)
    sse = {}
    
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(normalized_rfm_split)
        sse[k] = kmeans.inertia_
    
    kn = KneeLocator(
                 x=list(sse.keys()), 
                 y=list(sse.values()), 
                 curve='convex', 
                 direction='decreasing'
                 )
    k = kn.knee + increment - decrement
    return k

def run_kmeans(df, increment=0, decrement=0):
    """Run KMeans clustering, including the preprocessing of the data with optimum k. 
    """
    
    normalized_rfm_split = preprocess(df)
    k = find_k(df, increment, decrement)
    kmeans = KMeans(n_clusters=k, 
                    random_state=1)
    kmeans.fit(normalized_rfm_split)
    #df['prediction'] = kmeans.predict(normalized_rfm_split)
    return df.assign(cluster=kmeans.labels_)

def silhouette_avg(df_scaled, labels):
        """Compute silhouette coefficient given dataframe and its labels
        """
        silhouette_avg = silhouette_score(df_scaled, labels)
        return silhouette_avg

#Step1
if menu_id == "subid31":
        st.header("RFM-Based KMeans Clustering")
        rfmupload = st.file_uploader("Upload a Dataset for RFM Analysis ", type = ("csv", "xlsx"))
        if rfmupload is None:
                st.write("Upload the dataset")
        else:
                rfmdata = pd.read_csv(rfmupload)

                numeric_cols = st.multiselect("Select Numeric Columns", options=[c for c in rfmdata.select_dtypes(include=[int, float]).columns.values] )
                #Step2
                if len(numeric_cols)>2:
                        rfmdata.set_index('customer_id', inplace=True)
                        data_cluster = rfmdata[numeric_cols]
                        # Step1.1 Change datatype to float and remove missing values
                        data_cluster = data_cluster.dropna()
                        data_cluster = data_cluster.astype(float)
                        #Step3 
                        data_cluster = data_cluster.iloc[0:2000, :] #minimize the dataframe for time taken
                        data_cluster_normalized = preprocess(data_cluster)
                        
                        col21, col22 =st.columns(2)
                        optimized_k = find_k(data_cluster)
                        with col21:
                                #Plot k elbow
                                #Extract Top K from graph 
                                st.write(f'The optimized value for k is {optimized_k}')

                        # No need for the user to see it, but good to be kept in backend commented      
                        # with col22: 
                        #         # silhouette analysis

                        #         range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
                        #         sa = []
                        #         for num_clusters in range_n_clusters:
                                
                        #                 # intialise kmeans
                        #                 kmeans = KMeans(n_clusters=num_clusters, max_iter=100)
                        #                 kmeans.fit(data_cluster_normalized)
                                        
                        #                 cluster_labels = kmeans.labels_
                                        
                        #                 # silhouette score
                        #                 silhouette_avg = silhouette_score(data_cluster_normalized, cluster_labels)
                        #                 sa.append(silhouette_avg) 
                                        
                        #                 st.write(f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}")
                        
                        #Step5: After optimizing k, cluster the data
                        labeled_data = run_kmeans(data_cluster, increment=0, decrement=0)
                        silhouette_coef = silhouette_avg(data_cluster_normalized,labeled_data['cluster'] )
                        silhouette_coef = "{:.2f}".format(silhouette_coef)
                        st.write(f"The silhoutte coefficient for {optimized_k} is {silhouette_coef}")
                       
                        
                #2d or 3d Plot
                
                        if len(numeric_cols)==3:
                                #3d Plot
                                # 3d scatterplot using plotly
                                Scene = dict(xaxis = dict(title  = 'Recency'),yaxis = dict(title  = 'Frequency'),zaxis = dict(title  = 'Monetary'))

                                # model.labels_ is nothing but the predicted clusters i.e y_clusters
                                kmeans = KMeans(n_clusters=optimized_k, max_iter=100)
                                kmeans.fit(data_cluster_normalized)
                                labels = kmeans.labels_ 

                                trace = go.Scatter3d(x=data_cluster_normalized[:, 0], y=data_cluster_normalized[:, 1], z=data_cluster_normalized[:, 2], mode='markers',marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))
                                layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 600,width = 1000)
                                data = [trace]
                                fig = go.Figure(data = data, layout = layout)
                                # fig.show()
                                # st.plotly_chart
                                st.plotly_chart(fig)

                        #elif len(numeric_cols)==2:
                        #2d plot 
                        #due to time constraint I wont code it but the point is that I thought about it 

                        

                        #Step6: Show a dataframe of each labeled cluster with the aggregated mean for monetary, frequncy, recency
                        aggregated_data = labeled_data.groupby('cluster', as_index=False).agg(
                                                        recency=('recency','mean'),
                                                        frequency=('frequency','mean'),
                                                        monetary=('monetary','mean'),
                                                        cluster_size=('cluster','count')
                                                        ).round(1).sort_values(by='recency')
                        if optimized_k==1:
                                label1 = st.text_input("Enter the suitable label for cluster 0")
                                segments = { 0: label1}

                        elif optimized_k ==2:
                                label2_1 = st.text_input("Enter the suitable label for the first cluster 0")
                                label2_2 = st.text_input("Enter the suitable label for the second cluster 1")
                                segments = { 0:label2_1,1:label2_2}

                        elif optimized_k==3:
                                label3_1 = st.text_input("Enter the suitable label for the first cluster 0")
                                label3_2 = st.text_input("Enter the suitable label for the second cluster 1")
                                label3_3 = st.text_input("Enter the suitable label for the third cluster 2")
                                segments = { 0:label3_1,1:label3_2,2:label3_3}

                        elif optimized_k==4:
                                label4_1 = st.text_input("Enter the suitable label for the first cluster 0")
                                label4_2 = st.text_input("Enter the suitable label for the second cluster 1")
                                label4_3 = st.text_input("Enter the suitable label for the third cluster 2")
                                label4_4 = st.text_input("Enter the suitable label for the fourth cluster 3")
                                segments = { 0:label4_1, 1:label4_2, 2:label4_3, 3:label4_4 }

                        elif optimized_k==5:
                                label5_1 = st.text_input("Enter the suitable label for the first cluster 0")
                                label5_2 = st.text_input("Enter the suitable label for the second cluster 1")
                                label5_3 = st.text_input("Enter the suitable label for the third cluster 2")
                                label5_4 = st.text_input("Enter the suitable label for the fourth cluster 3")
                                label5_5 = st.text_input("Enter the suitable label for the fifth cluster 4")
                                segments = { 0:label5_1, 1:label5_2, 2:label5_3, 3:label5_1, 4:label5_5}

                        
                        #Labeling the clusters
                        labeled_data['segment'] = labeled_data['cluster'].map(segments)

                        plt1, plt2 =st.columns(2)
                        with plt1:
                                import plotly.graph_objects as go
                                
                                fig = go.Figure(data=[go.Table(
                                header=dict(values=list(aggregated_data.columns),
                                                fill_color='#d46262',
                                                align='left'),
                                cells=dict(values=[aggregated_data.cluster, aggregated_data.recency, aggregated_data.frequency, aggregated_data.monetary, aggregated_data.cluster_size],
                                        fill_color='#fce8e8',
                                        align='left'))
                                ])
                                st.plotly_chart(fig)
                        
                        with plt2:
                                import plotly.express as px
                                fig = px.pie(aggregated_data, values='cluster_size', names='cluster', color_discrete_sequence=px.colors.sequential.OrRd)
                                st.plotly_chart(fig)
                                
                        labeled_data.reset_index(inplace=True)
                        labeled_data = labeled_data.rename(columns = {'index':'customer_id'})
                        st.write(labeled_data.head())


                        @st.cache
                        def convert_df(df):
                                return df.to_csv().encode('utf-8')

                        labelsdf = convert_df(labeled_data)

                        st.download_button(
                                "Press to Download The Labeled Dataset",
                                labelsdf,
                                f"{company_name}_customerslabels.csv",
                                "text/csv",
                                key='download-csv'
                                )


#------------------------------------------
#------------Business Analytics------------
#------------------------------------------

if menu_id == "business_analytics":
        st.header("Business Analytics Dashboard")
        if fileuploadclean is not None:

                # Insert Filters 
                #Filter1: date 
                #Filter2: Customer_id
                #Filter3: Seller_id
                #Filter4: State

                max_year = pd.DatetimeIndex(dataclean['order_purchase_timestamp']).year.max()
                min_year = pd.DatetimeIndex(dataclean['order_purchase_timestamp']).year.min()
                max_month =  pd.DatetimeIndex(dataclean['order_purchase_timestamp']).month.max()
                min_month =  pd.DatetimeIndex(dataclean['order_purchase_timestamp']).month.min()
                max_day = pd.DatetimeIndex(dataclean['order_purchase_timestamp']).day.max()
                min_day = pd.DatetimeIndex(dataclean['order_purchase_timestamp']).day.min()
                
                
        
                f1, f2, f3, f4= st.columns(4)
                
                with f1:       
                        filterdate = st.checkbox('Filter by Date')
                        if filterdate:
                                start_date, end_date = st.date_input('Start Date  - End Date :',[pd.DatetimeIndex(dataclean['order_purchase_timestamp']).date.min(),pd.DatetimeIndex(dataclean['order_purchase_timestamp']).date.max()], disabled=False)
                        else:
                                start_date, end_date = st.date_input('Start Date  - End Date :',[pd.DatetimeIndex(dataclean['order_purchase_timestamp']).date.min(),pd.DatetimeIndex(dataclean['order_purchase_timestamp']).date.max()], disabled=True)
                with f2:
                        filterseller =st.checkbox('Filter by Seller')
                        if filterseller:
                                seller = st.selectbox("Seller Id", options=[c for c in dataclean['seller_id'].unique()], disabled=False)
                                seller_mask = (dataclean['seller_id']==seller)
                        else:
                                seller = st.selectbox("Seller Id", options=[c for c in dataclean['seller_id'].unique()], disabled=True)
                
                with f3:
                        filtercity = st.checkbox('Filter by City')
                        if filtercity:
                                city = st.selectbox("City",  options=[c for c in dataclean['customer_city'].unique()], disabled=False)
                                city_mask = (dataclean['customer_city']==city)
                        else:
                                city = st.selectbox("City",  options=[c for c in dataclean['customer_city'].unique()], disabled=True)

                with f4:
                        filtercat = st.checkbox('Filter by Category')
                        if filtercat:
                                categories = st.selectbox("Category", options =[c for c in dataclean['category'].unique()], disabled=False)
                                cat_mask = (dataclean['category']==categories)
                        else:
                               categories = st.selectbox("Category", options =[c for c in dataclean['category'].unique()], disabled=True) 

                if start_date > end_date:
                        st.error('Error: End date must fall after start date')
                else:
                        mask = (pd.DatetimeIndex(dataclean['order_purchase_timestamp']).date > start_date) & (pd.DatetimeIndex(dataclean['order_purchase_timestamp']).date  <= end_date)


                #filtered_values = np.where( (mask) & (seller_mask) & (city_mask) & (cat_mask))
                
                dataclean_filter =dataclean
                if filterdate == True:
                        dataclean_filter = dataclean.loc[mask]
               
                elif filterseller ==True:
                        dataclean_filter =dataclean.loc[seller_mask]

                elif filtercity ==True:
                        dataclean_filter =dataclean.loc[city_mask]

                elif filtercat ==True:
                        dataclean_filter =dataclean.loc[cat_mask]
                
                elif filterdate ==True and filterseller == True:
                        #dataclean_filter = dataclean.loc[mask] 
                        #dataclean_filter = dataclean.loc[seller_mask]
                        dataclean_filter = dataclean[(dataclean.loc[mask] ) & (dataclean.loc[seller_mask])]

                elif filterdate==True and filtercity == True:
                        dataclean_filter = dataclean.loc[mask and city_mask]
                elif filterdate==True and filtercat == True:
                        dataclean_filter = dataclean.loc[mask and cat_mask]
                elif filterseller==True and filtercity ==True:
                         dataclean_filter = dataclean.loc[seller_mask and city_mask]

                elif filterseller==True and filtercat ==True:
                         dataclean_filter = dataclean.loc[seller_mask and cat_mask]

                elif filtercity==True and filtercat ==True:
                         dataclean_filter = dataclean.loc[city_mask and cat_mask]

                elif filterdate==True and filterseller==True and filtercity == True:
                        dataclean_filter = dataclean.loc[mask and seller_mask and city_mask]

                elif filterdate==True and filterseller==True and filtercat == True:
                        dataclean_filter = dataclean.loc[mask and seller_mask and cat_mask]

                elif filterdate==True and filtercity==True and filtercat== True:
                        dataclean_filter = dataclean.loc[mask and city_mask and cat_mask]

                elif  filterseller==True and filtercity==True and filtercat == True:
                        dataclean_filter = dataclean.loc[seller_mask and city_mask and cat_mask]

                elif  filterdate==True and filterseller==True and filtercity==True and filtercat == True:
                        dataclean_filter = dataclean.loc[mask and seller_mask and city_mask and cat_mask]     



          
                #Insert Kpis
                #Kpi1: Sales
                #kpi2: Sellers
                #kpi3: Customers
                #kpi4: Orders

                #kpis = st.multiselect("Pick your kpis", options=[c for c in data.columns.values])
                #kpis_no = len(kpis)
                #if kpis_no>0:
                        
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                with kpi1:
                        theme_override = {'bgcolor': '#fdeded','title_color': '#c81717', 'content_color': '000000', 'icon': 'far fa-chart-bar', 'icon_color': '#c81717'}
                        hc.info_card(title = 'Sales', content = dataclean_filter['price'].agg(sum),
                        theme_override=theme_override)
                        #kpi1.metric(str(dataclean['price'].agg(sum)), "$")
                
                with kpi2:
                        theme_override = {'bgcolor': '#fdeded','title_color': '#c81717', 'content_color': '000000', 'icon': '', 'icon_color': '#c81717'}
                        hc.info_card(title = 'Orders Count', content = dataclean_filter['order_id'].nunique(),
                        theme_override=theme_override)
                        #kpi2.metric(str(dataclean['order_id'].nunique()), " ")
                
                with kpi3:
                        theme_override = {'bgcolor': '#fdeded','title_color': '#c81717', 'content_color': '000000', 'icon': 'fa-solid fa-radar', 'icon_color': '#c81717'}
                        hc.info_card(title = 'Customers Count', content = dataclean_filter['customer_id'].nunique(),
                        theme_override=theme_override)
                        #kpi3.metric(str(dataclean['customer_id'].nunique()), " ")
                
                with kpi4:
                        theme_override = {'bgcolor': '#fdeded','title_color': '#c81717', 'content_color': '000000', 'icon': '', 'icon_color': '#c81717'}
                        hc.info_card(title = 'Sellers Count', content = dataclean_filter['seller_id'].nunique(),
                        theme_override=theme_override)
                        #kpi4.metric(str(dataclean['seller_id'].nunique()), " ")

                st.markdown("<hr/>", unsafe_allow_html=True)
        

                chart1, chart2 = st.columns(2)

                with chart1:
                        st.subheader("Sales Over time")
                        dataclean_filter['date']= pd.DatetimeIndex(dataclean_filter['order_purchase_timestamp']).date
                        price_df = dataclean_filter.groupby('date', as_index=False).sum()

                        import plotly.express as px
                        fig = px.line(price_df, x='date', y="price")
                        st.plotly_chart(fig)
                
                with chart2:
                        st.subheader("Orders by Categories")

                        category_df = dataclean_filter.groupby('category', as_index=False)['customer_id'].count()
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Bar(
                                x=category_df['customer_id'],
                                y=dataclean_filter['category'],
                                orientation='h', ))
                        fig.update_layout( xaxis={'categoryorder':'total descending'})
                        st.plotly_chart(fig)
                
                chrt3, chrt4 = st.columns(2)
                with chrt3:
                        st.subheader('Sellers Distribution by City')

                        seller_df = dataclean_filter.groupby('seller_city', as_index=False)['seller_id'].count()
                        fig = go.Figure(go.Bar(
                                x=seller_df['seller_city'],
                                y=seller_df['seller_id'],
                         ))
                        fig.update_layout( xaxis={'categoryorder':'total descending'})
                        st.plotly_chart(fig)
                
                with chrt4:
                        import plotly.graph_objects as go

                        order_df = dataclean_filter.groupby('order_status', as_index=False)['order_id'].count()
                        labels = order_df['order_status']
                        values = order_df['order_id']

                        # pull is given as a fraction of the pie radius
                        fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])
                        st.plotly_chart(fig)







#----------------------------------------------------
#----------NLP Reviews Sentiment Analysis------------
#----------------------------------------------------

#Step1: select the columns to create the dataset to be used
#Step2: Clean text
#Step3: detect language 
#Step4: Translate to english 
#Step5: Sentiment Analysis

import re
import numpy as np

if menu_id == "subid32":
        st.header("NLP Reviews Sentiment Analysis")
        if fileupload is not None:

                #Step1 
                rev_col = st.multiselect("Select Reviews Text Column", options=[c for c in data.columns.values] )
                if len(rev_col)>0:  
                              
                        nlp_data = data[rev_col]
                        reviews_list = list(nlp_data[rev_col].values)
                        nlp_data[rev_col]=nlp_data[rev_col].astype("string")
                        if st.checkbox("Show Reviews Dataset"):
                                st.dataframe(nlp_data.head())
                                st.write(nlp_data.iloc[1])
                        
                        # rev_lst = str(reviews_list)
                        # st.write(rev_lst)
                
                        #nlp_data = nlp_data.reset_index()  # make sure indexes pair with number of rows
                        reviews_string = []
                        for index, row in nlp_data.iterrows():
                                reviews_string.append(row[rev_col])
                        
                        #st.write(str(reviews_string))

                        reviews_no = len(reviews_string)
                #Step2: Basic Text Cleaning (took 3 hours just because of datatypes problems)
                        def re_lineseparator(text_list):
                                """
                                replace	line separations with space (\r\n --> line separators)
                                """
                                return [re.sub('[\n\r]', ' ', r) for r in text_list]

                        def re_special_chars(text_list):
                                        """
                                        Replace special characters with a space
                                        """
                                        #Source: https://www.w3schools.com/python/python_regex.asp
                                        return [re.sub('\W', ' ', r) for r in text_list]
                        
                        def re_links(text_list):
                                        """
                                        Remove websites by replacing each link with space
                                        """
                                        #define link pattern
                                        #Source: http://urlregex.com/
                                        pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                                        return [re.sub(pattern, ' ', r) for r in text_list]
                        
                        def re_numbers(text_list):
                                        """
                                        replace numbers with space since we have the review score, any number included will not be of interest (most probably)
                                        """
                                        return [re.sub('[0-9]+', ' ', r) for r in text_list]
                        
                        def re_extrawhitespace(text_list):
                                        """
                                        Remove extra white space
                                        """
                                        #Source: https://www.codegrepper.com/code-examples/python/remove+extra+whitespace+from+string+python
                                        white_space = [re.sub(' +', ' ', r) for r in text_list]
                                        return white_space

                        for i in range(1,reviews_no):
                                reviews_string[i]= re_special_chars(reviews_string[i])
                                reviews_string[i]= re_lineseparator(reviews_string[i])
                                reviews_string[i]= re_links(reviews_string[i])
                                reviews_string[i]= re_numbers(reviews_string[i])
                                reviews_string[i]= re_extrawhitespace(reviews_string[i])
                                #reviews_clean_list.append(reviews_string)
                        
                        # clean_reviews_df = pd.DataFrame()
                        # for i in range(0,len(reviews_string)):
                        #          clean_reviews_df.iloc[i,]=reviews_string[i]
                                
                        # st.write(clean_reviews_df)

                        #Language Detector
                        lang_input = st.text_input("Insert a sentence to detect the language")
                        
                       

#---------------------------------------
#------------Personal Page--------------
#---------------------------------------
   
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image


if menu_id=="Contact Info":
        
        def load_lottieurl(url):
                r = requests.get(url)
                if r.status_code != 200:
                        return None
                        return r.json()


        # Use local CSS
        def local_css(file_name):
                with open(file_name) as f:
                 st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


        local_css("style/style.css")

        # ---- LOAD ASSETS ----
        lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_0yfsb3a1.json")
       

        # ---- HEADER SECTION ----
        with st.container():
                st.subheader("Hello, I am Amir")
                st.title("A Business and Data Analyst")
                st.write("[My portfolio >](https://amirbazzi859457.invisionapp.com/console/share/V8BS2J5RPEF)")

        # ---- WHY ME --- 
        with st.container():
                st.write("---")
                left_column, right_column = st.columns(2)
        with left_column:
                st.header("Why Hire Me")
                st.write("##")
                st.write(
                """
                I solve business problems with data, how?
                - I frame the problem and pin-point the bottleneck.
                - I collect the data needed and prepare it.
                - I build machine learning and forecasting models to predict and cluster.
                - I leverage the whole analysis process with business intelligence tools to communicate my findings. 
                """
                )
        with right_column:
                st_lottie(lottie_coding, height=300, key="coding")

        # ---- CONTACT ----
        with st.container():
                st.write("---")
                st.header("For More Info, Contact Me")
                st.write("##")

        contact_form = """
        <form action="https://formsubmit.co/arb11@mail.aub.edu" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Your name" required>
                <input type="email" name="email" placeholder="Your email" required>
                <textarea name="message" placeholder="Your message here" required></textarea>
                <button type="submit">Send</button>
        </form>
        """
        left_column, right_column = st.columns(2)
        with left_column:
                st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
                st.empty()


if menu_id=="Home":
        
        def load_lottieurl(url):
                r = requests.get(url)
                if r.status_code != 200:
                        return None
                        return r.json()


        # Use local CSS
        def local_css(file_name):
                with open(file_name) as f:
                 st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


        local_css("style/style.css")

        # ---- LOAD ASSETS ----
        lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_0yfsb3a1.json")
       

        # ---- HEADER SECTION ----
        with st.container():
                st.subheader(f"{company_name} E-commerce Marketplace")
                st.title("Business and Customer Analysis Tool")
                
        # ---- OLIST --- 
        with st.container():
                st.write("---")
                left_column, right_column = st.columns(2)
        with left_column:
                st.header("Interested in Analysing Data But Not a programmer? You got the right tool between your hands")
                st.write("##")
                st.write(
                """
                In this webapp, the user will be able to: 
                - Apply basic data cleaning and preprocessing steps, and download the cleaned file.
                - Inspect the data in general or detailed manner as upon his interest.
                - Apply unsupervised machine learning to cluster the customers according to different features and RFM especially
                - Clean the reviews text.( Perform sentiment analysis, coming soon . . . )
                - Perform Market Basket Analysis (coming soon . . . )
                - Make recommendations for the user based on interactions (collaborative-based recommender system) ((coming soon . . . ))
                """
                )

        with right_column:
                st_lottie(lottie_coding, height=300, key="coding")
