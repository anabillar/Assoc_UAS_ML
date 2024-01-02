import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

df["month"] = df['Order Date'].dt.month
df["day"] = df['Order Date'].dt.strftime("%A") 

st.title("Supermart Grocery Sales")

def get_recommendations(customer, category):
    customer_data = df[df["Customer Name"].str.lower() == customer.lower()]

    category_counts = customer_data["Sub Category"].value_counts()
    recommendations = category_counts[category_counts.index != category].head(1)

    return recommendations.index[0] if not recommendations.empty else None

def user_input_features():
    Customer = st.selectbox("Customer", ["Harish", "Sudha", "Hussain", "Jackson", "Ridhesh", "Sudeep", "Alan", "Sudeep", "Ravi", "Peer", "Ganesh"])
    SubCategory = st.selectbox("Sub Category", ["Health Drinks", "Soft Drinks", "Cookies", "Breads & Buns", "Noodles", "Chocolates", "Masalas", "Rice", "Biscuits", "Rice", "Atta & Flour"])
    return Customer, SubCategory

Customer, SubCategory = user_input_features()

if st.button('Submit'):
    recommendation = get_recommendations(Customer, SubCategory)
    
    if recommendation:
        st.success(f"Jika *{Customer}* membeli *{SubCategory}*, maka ia juga membeli *{recommendation}*")
    else:
        st.info("Rekomendasi tidak tersedia untuk kriteria yang dimasukkan.")

    plt.figure(figsize=(12, 6))
    selected_product_count = df[df["Sub Category"] == SubCategory]["Sub Category"].value_counts()
    recommended_product_count = df[df["Sub Category"] == recommendation]["Sub Category"].value_counts()

    plt.bar(selected_product_count.index, selected_product_count.values, alpha=0.7, label=f"{SubCategory}")
    plt.bar(recommended_product_count.index, recommended_product_count.values, alpha=0.7, label=f"{recommendation}")

    plt.xlabel('Sub Category')
    plt.ylabel('Frequency')
    plt.title('Frequency of Selected Category and Recommendation')
    plt.legend()
    st.pyplot(plt)
