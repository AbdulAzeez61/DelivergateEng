import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Create connection string
DATABASE_URL = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

def clean_customer_data(df):
    """Clean and validate customer data while preserving all records"""
    print("\nCleaning customer data...")
    
    # Store original row count
    original_count = len(df)
    
    # Make a copy to avoid modifying original data
    df = df.copy()
    
    # Report on null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("\nNull values found:")
        print(null_counts[null_counts > 0])
    
    # Clean customer_id - preserve but flag invalid IDs
    df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
    invalid_ids = df[df['customer_id'].isnull()]
    if not invalid_ids.empty:
        print(f"Found {len(invalid_ids)} invalid customer IDs - preserving records")
    
    # Handle missing names - preserve but mark as unknown
    df['name'] = df['name'].fillna('Unknown Customer')
    
    # Handle missing emails - create placeholder
    df['email'] = df['email'].fillna(df['customer_id'].astype(str) + '@placeholder.com')
    
    # Clean and standardize non-null names
    df['name'] = df['name'].apply(lambda x: x.strip().title() if pd.notna(x) else x)
    
    # Clean and standardize non-null emails
    df['email'] = df['email'].apply(lambda x: x.strip().lower() if pd.notna(x) else x)
    
    # Report cleaning results
    final_count = len(df)
    print(f"\nCustomer data cleaning summary:")
    print(f"Total records: {final_count}")
    print(f"Records with null values preserved: {null_counts.sum()}")
    
    return df

def clean_order_data(df):
    """Clean and validate order data"""
    print("\nCleaning order data...")
    
    # Store original row count
    original_count = len(df)
    
    # Make a copy to avoid modifying original data
    df = df.copy()
    
    # Check for duplicates
    duplicate_orders = df[df.duplicated(subset=['display_order_id'], keep=False)]
    if not duplicate_orders.empty:
        print(f"Found {len(duplicate_orders)} duplicate order IDs")
        df = df.drop_duplicates(subset=['display_order_id'], keep='first')
    
    # Handle missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("\nNull values found in orders:")
        print(null_counts[null_counts > 0])
        
        # Only drop orders with missing critical data
        critical_nulls = df[df['display_order_id'].isnull() | 
                          df['total_amount'].isnull() | 
                          df['created_at'].isnull()]
        if not critical_nulls.empty:
            print(f"Found {len(critical_nulls)} orders with missing critical data")
            df = df.dropna(subset=['display_order_id', 'total_amount', 'created_at'])
    
    # Clean numeric fields
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
    df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
    
    # Handle invalid amounts
    df.loc[df['total_amount'] < 0, 'total_amount'] = 0
    
    # Clean and standardize order IDs
    df['display_order_id'] = df['display_order_id'].str.strip().str.upper()
    
    # Convert dates
    try:
        df['created_at'] = pd.to_datetime(df['created_at'], format='%y-%m-%d %H:%M')
    except:
        try:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        except:
            print("Error converting dates - please check date format")
    
    # Remove future dates
    future_orders = df[df['created_at'] > datetime.now()]
    if not future_orders.empty:
        print(f"Found {len(future_orders)} orders with future dates")
        df = df[df['created_at'] <= datetime.now()]
    
    # Report cleaning results
    final_count = len(df)
    print(f"\nOrder data cleaning summary:")
    print(f"Original records: {original_count}")
    print(f"Final records: {final_count}")
    print(f"Removed records: {original_count - final_count}")
    
    return df


def import_data():
    try:
        print("Starting data import process...")
        
        # Create database engine
        engine = create_engine(DATABASE_URL)
        
        # Read CSV files
        print("Reading CSV files...")
        customers_df = pd.read_csv('data/customers.csv')
        orders_df = pd.read_csv('data/order.csv')
        
        # Clean the data
        customers_df = clean_customer_data(customers_df)
        orders_df = clean_order_data(orders_df)
        
        
        # Import to database
        print("\nImporting data to database...")
        with engine.connect() as conn:
            customers_df.to_sql('customers', conn, if_exists='replace', index=False)
            print("Customers imported successfully!")
            
            orders_df.to_sql('orders', conn, if_exists='replace', index=False)
            print("Orders imported successfully!")
        
        
        print("\nData import completed successfully!")
        
    except Exception as e:
        print(f"Error during import: {str(e)}")

if __name__ == "__main__":
    import_data()