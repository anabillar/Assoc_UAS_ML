import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.frequent_patterns import association_rules, apriori

# Memuat data
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce', infer_datetime_format=True)

df["month"] = df['Order Date'].dt.month
df["day"] = df['Order Date'].dt.strftime("%A") 

st.title("Supermart Grocery Sales")

def get_data(month='', day='', Customer=''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].astype(str) == month) & 
        (data["Customer Name"].str.lower() == Customer.lower()) &  
        (data["day"] == day) 
    ]
    return filtered

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_product_df(product_antecedents):
    data_rules = rules[["antecedents", "consequents"]].copy()

    data_rules["antecedents"] = data_rules["antecedents"].apply(parse_list)
    data_rules["consequents"] = data_rules["consequents"].apply(parse_list)

    return list(data_rules.loc[data_rules["antecedents"] == product_antecedents].iloc[0,:])

def user_input_features():
    Customer = st.selectbox("Customer", ["Harish", "Sudha", "Hussain", "Jackson", "Ridhesh", "Sudeep", "Alan", "Sudeep", "Ravi", "Peer", "Ganesh"])
    SubCategory = st.selectbox("Sub Category", ["Health Drinks", "Soft Drinks", "Cookies", "Breads & Buns", "Noodles", "Chocolates", "Masalas", "Rice", "Biscuits", "Rice", "Atta & Flour"])
    month = st.select_slider("Month", [str(i) for i in range(1, 13)])
    day = st.select_slider("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], value="Monday")

    return month, day, SubCategory, Customer

month, day, SubCategory, Customer = user_input_features()

if st.button('Submit'):
    data = get_data(month.lower(), day, Customer)

    if not data.empty:
        cat_counts = df.groupby(["Customer", "Sub Category"])["Customer"].count().reset_index(name="Count")
        cat_count_pivot = cat_counts.pivot_table(index='Order ID', columns='Sub Category', values='Count', aggfunc='sum').fillna(0)
        cat_count_pivot = cat_count_pivot.applymap(encode)

        support = 0.01
        frequent_cat = apriori(cat_count_pivot, min_support=support, use_colnames=True)

        metric = "lift"
        min_threshold = 1

        rules = association_rules(frequent_cat, metric='lift', min_threshold=1).sort_values('lift', ascending=False).reset_index(drop=True)[["antecedents", "consequents", "support", "confidence", "lift"]]
        rules.sort_values('confidence', ascending=False, inplace=True)

        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        x = rules['support']
        y = rules['confidence']
        z = rules['lift']

        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_zlabel("Lift")

        ax.scatter(x, y, z)
        ax.set_title("3D Distribution of Association Rules")

        st.pyplot(fig)  # Menampilkan plot di Streamlit

        st.write("Hasil Analisis:")
        st.write(data)  

        st.markdown("Rekomendasi: ")
        st.success(f"Jika *{Customer}* membeli *{SubCategory}, maka ia juga membeli *{return_product_df(SubCategory)[1]}**")
    else:
        st.write("Data tidak tersedia.")
