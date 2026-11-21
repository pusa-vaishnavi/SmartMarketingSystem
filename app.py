
# ============================================================
# SMART MARKETING DECISION SUPPORT SYSTEM
# Purchase, Return & Cancellation Pattern Analysis
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Smart Marketing System",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
h1, h2, h3 {
    color: #1e3a5f;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Smart Marketing Decision Support System")
st.caption(
    "Customer purchase behaviour, product returns, "
    "cancellation analysis and promotion intelligence"
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Customer Segmentation",
        "Product Intelligence",
        "Marketing Recommendations"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Retail Dataset",
    type=["csv", "xlsx"]
)

# ---------------- DATA LOADING ----------------

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)

    return data

if uploaded_file is None:
    st.info("Upload your Online Retail CSV or Excel dataset to begin.")

    st.markdown("""
    ### Application features
    - Sales and customer analytics
    - RFM customer segmentation
    - PCA visualization
    - K-Means clustering
    - Return and cancellation analysis
    - Marketing campaign suggestions
    """)

    st.stop()

df = load_data(uploaded_file)

# ---------------- DATA CLEANING ----------------

required_columns = [
    "InvoiceNo", "StockCode", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error("Missing required columns: " + str(missing_columns))
    st.stop()

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"], errors="coerce"
)

df["CustomerID"] = pd.to_numeric(
    df["CustomerID"], errors="coerce"
)

df.dropna(
    subset=["InvoiceDate", "Quantity", "UnitPrice"],
    inplace=True
)

df = df[
    (df["Quantity"] != 0) &
    (df["UnitPrice"] >= 0)
].copy()

df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df["CustomerID"] = df["CustomerID"].astype("Int64")

df["TotalAmount"] = (
    df["Quantity"] * df["UnitPrice"]
)

# ---------------- TRANSACTION CLASSIFICATION ----------------

df["IsCancellation"] = df["InvoiceNo"].str.startswith("C")

# Negative quantities that are not cancellation invoices
df["IsReturn"] = (
    (df["Quantity"] < 0) &
    (~df["IsCancellation"])
)

df["IsSale"] = (
    (df["Quantity"] > 0) &
    (~df["IsCancellation"])
)

sales = df[df["IsSale"]].copy()
returns = df[df["IsReturn"]].copy()
cancellations = df[df["IsCancellation"]].copy()

# ---------------- RFM ANALYSIS ----------------

customer_data = sales.dropna(
    subset=["CustomerID"]
).copy()

reference_date = customer_data["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = customer_data.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x:
        (reference_date - x.max()).days
    ),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalAmount", "sum")
).reset_index()

# ---------------- RFM SCORES ----------------

rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    4,
    labels=[4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    4,
    labels=[1, 2, 3, 4]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    4,
    labels=[1, 2, 3, 4]
).astype(int)

rfm["RFM_Score"] = (
    rfm["R_Score"] +
    rfm["F_Score"] +
    rfm["M_Score"]
)

# ---------------- K-MEANS ----------------

features = rfm[[
    "Recency", "Frequency", "Monetary"
]].copy()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

n_clusters = min(4, len(rfm))

if n_clusters < 2:
    st.error("At least two customers are required for clustering.")
    st.stop()

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(scaled_features)

# ---------------- PCA ----------------

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

rfm["PCA1"] = pca_features[:, 0]
rfm["PCA2"] = pca_features[:, 1]

# ---------------- SEGMENT LABELS ----------------

cluster_summary = rfm.groupby("Cluster").agg(
    AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    Customers=("CustomerID", "count")
).reset_index()

# Rank clusters using their frequency and spending
cluster_summary["ValueRank"] = (
    cluster_summary["AvgFrequency"].rank(pct=True) +
    cluster_summary["AvgMonetary"].rank(pct=True)
)

ranked_clusters = cluster_summary.sort_values(
    "ValueRank", ascending=False
)["Cluster"].tolist()

cluster_names = [
    "Champions",
    "Loyal Customers",
    "Potential Customers",
    "At-Risk Customers"
]

cluster_map = {
    cluster: cluster_names[i]
    for i, cluster in enumerate(ranked_clusters)
}

rfm["Segment"] = rfm["Cluster"].map(cluster_map)

# ---------------- PRODUCT ANALYSIS ----------------

product_col = "Description" if "Description" in df.columns else "StockCode"

product_sales = sales.groupby(product_col).agg(
    SalesQuantity=("Quantity", "sum"),
    Revenue=("TotalAmount", "sum"),
    Orders=("InvoiceNo", "nunique")
).reset_index()

product_returns = returns.groupby(product_col).agg(
    ReturnedQuantity=("Quantity", lambda x: abs(x.sum())),
    ReturnTransactions=("InvoiceNo", "nunique")
).reset_index()

product_cancellations = cancellations.groupby(product_col).agg(
    CancelledQuantity=("Quantity", lambda x: abs(x.sum())),
    CancellationTransactions=("InvoiceNo", "nunique")
).reset_index()

product_analysis = product_sales.merge(
    product_returns, on=product_col, how="left"
).merge(
    product_cancellations, on=product_col, how="left"
)

product_analysis = product_analysis.fillna(0)

product_analysis["ReturnRate"] = (
    product_analysis["ReturnedQuantity"] /
    (product_analysis["SalesQuantity"] +
     product_analysis["ReturnedQuantity"] + 1e-9)
) * 100

product_analysis["CancellationRate"] = (
    product_analysis["CancellationTransactions"] /
    (product_analysis["Orders"] +
     product_analysis["CancellationTransactions"] + 1e-9)
) * 100

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.header("Marketing Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Revenue",
        f"£{sales['TotalAmount'].sum():,.2f}"
    )

    c2.metric(
        "Customers",
        f"{sales['CustomerID'].nunique():,}"
    )

    c3.metric(
        "Sales Invoices",
        f"{sales['InvoiceNo'].nunique():,}"
    )

    c4.metric(
        "Cancellation Invoices",
        f"{cancellations['InvoiceNo'].nunique():,}"
    )

    st.subheader("Revenue Trend")

    revenue_trend = sales.groupby(
        sales["InvoiceDate"].dt.to_period("M")
    )["TotalAmount"].sum().reset_index()

    revenue_trend["InvoiceDate"] = (
        revenue_trend["InvoiceDate"].astype(str)
    )

    fig = px.line(
        revenue_trend,
        x="InvoiceDate",
        y="TotalAmount",
        markers=True,
        title="Monthly Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Segments")

    segment_counts = rfm["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Customers"]

    fig = px.pie(
        segment_counts,
        names="Segment",
        values="Customers",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Selling Products")

    top_products = product_sales.sort_values(
        "Revenue", ascending=False
    ).head(10)

    fig = px.bar(
        top_products,
        x="Revenue",
        y=product_col,
        orientation="h",
        title="Top 10 Products by Revenue"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- CUSTOMER SEGMENTATION ----------------

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    st.write("Customers grouped using RFM and K-Means.")

    st.dataframe(
        rfm[[
            "CustomerID", "Recency", "Frequency",
            "Monetary", "RFM_Score", "Segment"
        ]],
        use_container_width=True
    )

    fig = px.scatter(
        rfm,
        x="PCA1",
        y="PCA2",
        color="Segment",
        hover_data=["CustomerID", "Monetary", "Frequency"],
        title="Customer Clusters using PCA"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Summary")

    st.dataframe(
        rfm.groupby("Segment").agg(
            Customers=("CustomerID", "count"),
            AverageRecency=("Recency", "mean"),
            AverageOrders=("Frequency", "mean"),
            AverageSpending=("Monetary", "mean")
        ).round(2),
        use_container_width=True
    )

# ---------------- PRODUCT INTELLIGENCE ----------------

elif page == "Product Intelligence":

    st.header("Product Return & Cancellation Intelligence")

    st.subheader("Products with Highest Return Rates")

    high_returns = product_analysis.sort_values(
        "ReturnRate", ascending=False
    ).head(10)

    fig = px.bar(
        high_returns,
        x="ReturnRate",
        y=product_col,
        orientation="h",
        title="Top Return Rate Products"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Products with Highest Cancellation Activity")

    high_cancellations = product_analysis.sort_values(
        "CancellationTransactions", ascending=False
    ).head(10)

    fig = px.bar(
        high_cancellations,
        x="CancellationTransactions",
        y=product_col,
        orientation="h",
        title="Cancellation Transactions"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Complete Product Analysis")

    st.dataframe(
        product_analysis.sort_values(
            "Revenue", ascending=False
        ),
        use_container_width=True
    )

# ---------------- MARKETING RECOMMENDATIONS ----------------

elif page == "Marketing Recommendations":

    st.header("Smart Marketing Recommendation Engine")

    st.write(
        "Recommendations are generated from customer segments, "
        "sales, return rates and cancellation activity."
    )

    st.subheader("Customer Campaign Recommendations")

    for segment in rfm["Segment"].unique():

        count = (rfm["Segment"] == segment).sum()

        if segment == "Champions":
            action = "Reward with loyalty benefits and exclusive offers."

        elif segment == "Loyal Customers":
            action = "Offer bundles and referral rewards."

        elif segment == "Potential Customers":
            action = "Use personalized promotions to encourage repeat purchases."

        else:
            action = "Try re-engagement campaigns and feedback surveys."

        st.info(
            f"{segment}: {count} customers. {action}"
        )

    st.subheader("Product Marketing Recommendations")

    for _, row in product_analysis.iterrows():

        if row["SalesQuantity"] <= 0:
            continue

        if row["ReturnRate"] >= 20:
            st.warning(
                f"{row[product_col]}: High return rate "
                f"({row['ReturnRate']:.1f}%). "
                "Review product quality or description before promotion."
            )

        elif row["Revenue"] > product_analysis["Revenue"].median():
            st.success(
                f"{row[product_col]}: Strong revenue. "
                "Consider a targeted promotion or product bundle."
            )

    st.subheader("Download Customer Segmentation")

    csv_data = rfm.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Customer Analysis CSV",
        data=csv_data,
        file_name="customer_segmentation.csv",
        mime="text/csv"
    )

    st.subheader("Download Product Analysis")

    product_csv = product_analysis.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Product Analysis CSV",
        data=product_csv,
        file_name="product_intelligence.csv",
        mime="text/csv"
    )

# ---------------- END ----------------

st.sidebar.success("Analysis completed successfully.")
