import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime
from config import DATABASE_URL

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.figure_factory as ff 

# Initialize database connection
@st.cache_resource
def init_connection():
    try:
        return create_engine(DATABASE_URL)
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None

engine = init_connection()

# Cached data loading
@st.cache_data
def get_data(query, params=None):
    if engine is None:
        st.error("Database connection not initialized")
        return pd.DataFrame()
    
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return pd.DataFrame()

# Data loading for ML model
def load_customer_data():
    query = """
    SELECT customer_id, COUNT(order_id) AS total_orders, 
           SUM(total_amount) AS total_revenue
    FROM orders
    GROUP BY customer_id
    """
    customer_data = get_data(query)
    customer_data['is_repeat'] = (customer_data['total_orders'] > 1).astype(int)
    return customer_data

# Training ML model on retrieved data
def train_model(data):
    if len(data) < 50:
        st.warning("Not enough data for reliable model training.")
        return None, None

    X = data[['total_orders', 'total_revenue']]
    y = data['is_repeat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy, precision, recall, conf_matrix, scaler


def main():
    st.title("Delivergate Order Analytics Dashboard")
    
    if engine is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    try:
        # Get date range for orders
        all_orders = get_data("SELECT MIN(order_date) as min_date, MAX(order_date) as max_date FROM orders")
        if all_orders.empty:
            st.error("No orders found in database")
            st.stop()
            
        min_date = pd.to_datetime(all_orders['min_date'].iloc[0])
        max_date = pd.to_datetime(all_orders['max_date'].iloc[0])
        
        # Date range filter
        start_date, end_date = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        
        # Ensure end_date is inclusive by setting it to end of day
        end_date = datetime.combine(end_date, datetime.max.time())
        
        # Total amount spent filter
        max_amount = get_data("SELECT MAX(total_amount) as max_amount FROM orders")['max_amount'].iloc[0]
        total_spent = st.sidebar.slider(
            "Minimum Total Amount Spent",
            min_value=0.0,
            max_value=float(max_amount),
            value=0.0,
            step=100.0
        )
        
        # Minimum orders filter
        max_orders = get_data(""" 
            SELECT MAX(order_count) as max_orders FROM (
                SELECT customer_id, COUNT(*) as order_count 
                FROM orders 
                GROUP BY customer_id
            ) t""")['max_orders'].iloc[0]
        
        min_orders = st.sidebar.number_input(
            "Minimum Number of Orders",
            min_value=1,
            max_value=int(max_orders),
            value=1
        )
        
        # Build query based on filters
        query = """
        WITH customer_stats AS (
            SELECT 
                c.customer_id,
                c.customer_name,
                COUNT(o.order_id) as order_count,
                SUM(o.total_amount) as total_spent
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            WHERE o.order_date BETWEEN :start_date AND :end_date
            GROUP BY c.customer_id, c.customer_name
            HAVING COUNT(o.order_id) >= :min_orders
            AND SUM(o.total_amount) >= :total_spent
        )
        SELECT 
            o.order_id,
            o.order_date,
            o.total_amount,
            c.customer_name as customer_name,
            cs.total_spent,
            cs.order_count
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN customer_stats cs ON c.customer_id = cs.customer_id
        WHERE o.order_date BETWEEN :start_date AND :end_date
        ORDER BY o.order_date DESC
        """
        
        # Load filtered data
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'min_orders': min_orders,
            'total_spent': total_spent
        }
        df = get_data(query, params=params)
        
        if df.empty:
            st.warning("No data found for the selected filters")
            st.stop()
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Revenue", f"${df['total_amount'].sum():,.2f}")
        with col2:
            st.metric("Unique Customers", len(df['customer_name'].unique()))
        with col3:
            st.metric("Total Orders", len(df))
        
        # Top 10 customers chart
        st.subheader("Top 10 Customers by Revenue")
        top_customers = (df.groupby('customer_name')['total_amount']
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index())
        
        fig_top_customers = px.bar(
            top_customers,
            x='customer_name',
            y='total_amount',
            title="Top 10 Customers by Revenue",
            labels={'customer_name': 'Customer', 'total_amount': 'Total Revenue ($)'}
        )
        fig_top_customers.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_top_customers, use_container_width=True)
        
        # Revenue over time
        st.subheader("Revenue Over Time")
        df['date'] = pd.to_datetime(df['order_date']).dt.date
        daily_revenue = df.groupby('date')['total_amount'].sum().reset_index()
        
        fig_revenue = px.line(
            daily_revenue,
            x='date',
            y='total_amount',
            title="Daily Revenue",
            labels={'date': 'Date', 'total_amount': 'Revenue ($)'}
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Display filtered data
        st.subheader("Filtered Orders")

        st.markdown(
            """
            <style>
            .dataframe-container table {
                width: 100% !important;
                overflow-x: hidden !important;
                word-wrap: break-word; 
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            df.sort_values('order_count', ascending=True),
            use_container_width=True,
            column_config={
                'order_date': st.column_config.DatetimeColumn('Order Date'),
                'total_amount': st.column_config.NumberColumn(
                    'Total Amount',
                    format='$%.2f'
                ),
                'total_spent': st.column_config.NumberColumn(
                    'Customer Total Spent',
                    format='$%.2f'
                )
            },
            hide_index=True
        )
    
        st.header("Customer Repeat Purchase Prediction Model")
        customer_data = load_customer_data()
        
        if not customer_data.empty:
            model, accuracy, precision, recall, conf_matrix, scaler = train_model(customer_data)
            
            if model:
                st.success(f"Model trained successfully with accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                
                st.subheader("Confusion Matrix")
                conf_matrix_df = pd.DataFrame(
                    conf_matrix,
                    index=["Actual Non-Repeat", "Actual Repeat"],
                    columns=["Predicted Non-Repeat", "Predicted Repeat"]
                )
                st.write(conf_matrix_df)
                
                # Heatmap Visualization of Confusion Matrix
                st.subheader("Confusion Matrix Heatmap")
                fig = ff.create_annotated_heatmap(
                    z=conf_matrix,
                    x=["Predicted Non-Repeat", "Predicted Repeat"],
                    y=["Actual Non-Repeat", "Actual Repeat"],
                    colorscale="Blues",
                    showscale=True
                )
                fig.update_layout(title="Confusion Matrix Heatmap")
                st.plotly_chart(fig)
                
                # User Input for Prediction
                st.subheader("Predict Repeat Purchase")
                total_orders_input = st.number_input("Total Orders", min_value=0, value=1)
                total_revenue_input = st.number_input("Total Revenue", min_value=0.0, value=100.0)
                
                if st.button("Predict"):
                    input_data = np.array([[total_orders_input, total_revenue_input]])
                    input_data_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_data_scaled)
                    
                    if prediction[0] == 1:
                        st.write("Prediction: This customer is likely a repeat purchaser.")
                    else:
                        st.write("Prediction: This customer is unlikely to be a repeat purchaser.")
                
                # Show sample predictions from the test set
                st.subheader("Sample Predictions from Test Set")
                X = customer_data[['total_orders', 'total_revenue']]
                y = customer_data['is_repeat']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_test_scaled = scaler.transform(X_test)
                test_predictions = model.predict(X_test_scaled)
                sample_results = pd.DataFrame({
                    "Total Orders": X_test['total_orders'],
                    "Total Revenue": X_test['total_revenue'],
                    "Actual": y_test,
                    "Predicted": test_predictions
                }).head(10)
                st.write(sample_results)
            else:
                st.error("Model training was unsuccessful.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your database connection and data integrity.")

if __name__ == "__main__":
    main()