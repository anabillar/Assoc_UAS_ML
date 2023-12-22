import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def apriori_analysis(df, min_support, min_confidence):
    te = TransactionEncoder()
    te_ary = te.fit(df).transform(df)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(
        df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return frequent_itemsets, rules


def main():
    st.title("Market Basket Analysis E-Commerce")

    # File upload
    uploaded_file = st.file_uploader(
        "Analytics case study (market basket analysis ecommerce).csv", type=["csv"])
    if not uploaded_file:
        st.info("Analytics case study (market basket analysis ecommerce).csv")

    # Parameters
    min_support = st.slider("Minimal Support", 0.0, 0.01, 0.02, 0.03)
    min_confidence = st.slider("Minimal Confidence", 0.1, 0.5, 0.7, 0.0, 0.6)

    # Load data
    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            st.header("Data Preview")
            st.write(df.head())

            # Run Apriori analysis
            frequent_itemsets, rules = apriori_analysis(
                df, min_support, min_confidence)

            # Display results
            st.header("Frequent Itemsets")
            st.write(frequent_itemsets)

            st.header("Association Rules")
            st.write(rules)


if _name_ == "_main_":
    main()