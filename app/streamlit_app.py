import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from config import DATABASE_URL

# Initialize database connection
@st.cache_resource
def init_connection():
    return create_engine(DATABASE_URL)

# Cached data loading
@st.cache_data
def get_data(query):
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

# Initialize connection
engine = init_connection()

def main():
    st.title("Delivergate Order Analytics Dashboard")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    all_orders = get_data("SELECT MIN(created_at) as min_date, MAX(created_at) as max_date FROM orders")
    min_date = pd.to_datetime(all_orders['min_date'].iloc[0])
    max_date = pd.to_datetime(all_orders['max_date'].iloc[0])
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    # Total amount spent filter
    total_spent = st.sidebar.slider(
        "Minimum Total Amount Spent",
        min_value=0,
        max_value=10000,
        value=0,
        step=100
    )
    
    # Minimum orders filter
    min_orders = st.sidebar.number_input(
        "Minimum Number of Orders",
        min_value=1,
        value=1
    )
    
    # Build query based on filters
    query = f"""
    WITH customer_stats AS (
        SELECT 
            c.customer_id,
            c.name,
            COUNT(o.id) as order_count,
            SUM(o.total_amount) as total_spent
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.created_at BETWEEN '{date_range[0]}' AND '{date_range[1]}'
        GROUP BY c.customer_id, c.name
        HAVING COUNT(o.id) >= {min_orders}
        AND SUM(o.total_amount) >= {total_spent}
    )
    SELECT 
        o.*,
        c.name as customer_name,
        cs.total_spent,
        cs.order_count
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN customer_stats cs ON c.customer_id = cs.customer_id
    WHERE o.created_at BETWEEN '{date_range[0]}' AND '{date_range[1]}'
    ORDER BY o.created_at DESC
    """
    
    # Load filtered data
    df = get_data(query)
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue", f"${df['total_amount'].sum():,.2f}")
    with col2:
        st.metric("Unique Customers", len(df['customer_id'].unique()))
    with col3:
        st.metric("Total Orders", len(df))
    
    # Top 10 customers chart
    st.subheader("Top 10 Customers by Revenue")
    top_customers = df.groupby('customer_name')['total_amount'].sum().sort_values(ascending=False).head(10)
    fig_top_customers = px.bar(
        top_customers,
        title="Top 10 Customers by Revenue",
        labels={'customer_name': 'Customer', 'total_amount': 'Total Revenue'}
    )
    st.plotly_chart(fig_top_customers)
    
    # Revenue over time
    st.subheader("Revenue Over Time")
    daily_revenue = df.groupby(pd.to_datetime(df['created_at']).dt.date)['total_amount'].sum().reset_index()
    fig_revenue = px.line(
        daily_revenue,
        x='created_at',
        y='total_amount',
        title="Daily Revenue",
        labels={'created_at': 'Date', 'total_amount': 'Revenue'}
    )
    st.plotly_chart(fig_revenue)
    
    # Display filtered data
    st.subheader("Filtered Orders")
    st.dataframe(df)

if __name__ == "__main__":
    main()