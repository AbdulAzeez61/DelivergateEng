import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from config import DATABASE_URL

def clean_customers_data(df):
    """Clean and prepare customers data"""
    # Remove duplicates based on email
    df = df.drop_duplicates(subset=['email'], keep='first')
    
    # Ensure customer_id is integer
    df['customer_id'] = pd.to_numeric(df['customer_id'])
    
    return df

def clean_orders_data(df):
    """Clean and prepare orders data"""
    # Convert created_at to datetime
    df['created_at'] = pd.to_datetime(df['created_at'], format='%y-%m-%d %H:%M')
    
    # Convert numeric columns
    df['id'] = pd.to_numeric(df['id'])
    df['total_amount'] = pd.to_numeric(df['total_amount'])
    df['customer_id'] = pd.to_numeric(df['customer_id'])
    
    return df

def import_data():
    try:
        # Create database connection
        engine = create_engine(DATABASE_URL)
        
        # Read CSV files
        customers_df = pd.read_csv('data/customers.csv')
        orders_df = pd.read_csv('data/orders.csv')
        
        # Clean data
        customers_df = clean_customers_data(customers_df)
        orders_df = clean_orders_data(orders_df)
        
        # Import to database
        with engine.connect() as conn:
            # Import customers first (due to foreign key constraint)
            customers_df.to_sql('customers', conn, if_exists='replace', index=False)
            
            # Import orders
            orders_df.to_sql('orders', conn, if_exists='replace', index=False)
            
        print("Data import completed successfully!")
        
    except Exception as e:
        print(f"Error during import: {str(e)}")

if __name__ == "__main__":
    import_data()