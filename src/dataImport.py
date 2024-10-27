import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def analyze_data_integrity(customers_df, orders_df):
    """Analyze relationships between customers and orders"""
    customer_ids = set(customers_df['customer_id'].unique())
    order_customer_ids = set(orders_df['customer_id'].unique())
    
    orphaned_customer_ids = order_customer_ids - customer_ids
    
    if orphaned_customer_ids:
        logger.warning(f"Found {len(orphaned_customer_ids)} customer IDs in orders that don't exist in customers table")
        orphaned_orders = orders_df[orders_df['customer_id'].isin(orphaned_customer_ids)]
        logger.warning(f"Number of orphaned orders: {len(orphaned_orders)}")
        
        # Save orphaned orders to CSV for review
        orphaned_orders.to_csv('orphaned_orders.csv', index=False)
        logger.info("Saved orphaned orders to 'orphaned_orders.csv'")
        
        # Basic statistics about orphaned orders
        logger.info("\nOrphaned Orders Statistics:")
        logger.info(f"Total amount involved: {orphaned_orders['total_amount'].sum():.2f}")
        logger.info(f"Date range: {orphaned_orders['created_at'].min()} to {orphaned_orders['created_at'].max()}")
        logger.info(f"Unique missing customer IDs: {len(orphaned_customer_ids)}")
        
    return orphaned_customer_ids

def clean_customer_data(df):
    """Clean and validate customer data"""
    try:
        initial_count = len(df)
        
        # Drop rows with missing customer_ids
        df = df.dropna(subset=['customer_id'])
        logger.info(f"Dropped {initial_count - len(df)} rows with missing customer IDs")
        
        # Remove duplicates but keep the first occurrence
        duplicates = df.duplicated(subset=['customer_id'], keep='first').sum()
        df = df.drop_duplicates(subset=['customer_id'], keep='first')
        if duplicates > 0:
            logger.warning(f"Removed {duplicates} duplicate customer IDs")
        
        # Fill NaN values in name with a placeholder
        df['name'] = df['name'].fillna('Unknown Customer')
        df['email'] = df['email'].fillna('Unknown Customer')
        df['email'] = df['email'].astype(str)

        # Convert customer_id to integer
        df['customer_id'] = df['customer_id'].astype(int)
        
        # Clean name
        df['name'] = df['name'].str.strip()
        df['email'] = df['email'].str.strip()

        df['name'] = df['name'].str.replace(r'[^\w\s-]', '', regex=True)
        df.loc[df['name'].str.len() == 0, 'name'] = 'Unknown Customer'
        df['name'] = df['name'].str[:255]
        
        logger.info(f"Cleaned {len(df)} customer records")
        return df
    except Exception as e:
        logger.error(f"Error cleaning customer data: {str(e)}")
        raise

def clean_order_data(df, valid_customer_ids=None):
    """Clean and validate order data"""
    try:
        initial_count = len(df)
        
        # Drop orders with missing order_ids
        df = df.dropna(subset=['id'])
        df = df.dropna(subset=['display_order_id'])
        df = df.dropna(subset=['customer_id'])

        logger.info(f"Dropped {initial_count - len(df)} orders with missing order IDs and id's")
        
        # Remove duplicates
        duplicates = df.duplicated(subset=['id'], keep='first').sum()
        df = df.drop_duplicates(subset=['display_order_id'], keep='first')
        if duplicates > 0:
            logger.warning(f"Removed {duplicates} duplicate order IDs")
        
        # Convert IDs to integers
        df['id'] = df['id'].astype(int)
        df['display_order_id'] = df['display_order_id'].astype(str)
        df['customer_id'] = df['customer_id'].astype(int)
        
        # Filter for valid customer IDs if provided
        if valid_customer_ids is not None:
            invalid_orders = df[~df['customer_id'].isin(valid_customer_ids)]
            df = df[df['customer_id'].isin(valid_customer_ids)]
            logger.info(f"Filtered out {len(invalid_orders)} orders with invalid customer IDs")
        
        # Handle total_amount
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
        df['total_amount'] = df['total_amount'].fillna(0.0)
        
        # Handle created_at
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['created_at'] = df['created_at'].fillna(pd.Timestamp.now())
        
        logger.info(f"Cleaned {len(df)} order records")
        return df
    except Exception as e:
        logger.error(f"Error cleaning order data: {str(e)}")
        raise

def import_data_sqlalchemy(customer_file, order_file, connection_string, batch_size=1000):
    """Import data using mysql-connector-python"""
    try:
        engine = create_engine(connection_string)
        # Read and clean data
        logger.info("Reading and cleaning data...")
        customers_df = clean_customer_data(pd.read_csv(customer_file))
        orders_df = clean_order_data(pd.read_csv(order_file))
        
        orphaned_customer_ids = analyze_data_integrity(customers_df, orders_df)

        # Connect to database
        # conn = mysql.connector.connect(**db_config)
        # cursor = conn.cursor()
        
        # Import customers
        logger.info("Importing customers...")
        customer_query = "INSERT INTO customers (customer_id, name, email) VALUES (%s, %s,%s)"
        
        # for i in range(0, len(customers_df), batch_size):
        #     batch = customers_df.iloc[i:i + batch_size]
        #     values = list(batch.itertuples(index=False, name=None))
        #     cursor.executemany(customer_query, values)
        #     conn.commit()
        #     logger.info(f"Imported customers batch {i//batch_size + 1}")
        
        # # Import orders
        # logger.info("Importing orders...")
        # order_query = """
        #     INSERT INTO orders (id,display_order_id,total_amount,created_at,customer_id) 
        #     VALUES (%s, %s, %s, %s,%s)
        # """
        
        # for i in range(0, len(orders_df), batch_size):
        #     batch = orders_df.iloc[i:i + batch_size]
        #     values = [
        #         (row.id, row.display_order_id, row.total_amount, row.created_at,row.customer_id)
        #         for row in batch.itertuples(index=False)
        #     ]
        #     cursor.executemany(order_query, values)
        #     conn.commit()
        #     logger.info(f"Imported orders batch {i//batch_size + 1}")
        
        # logger.info("Data import completed successfully!")
        logger.info("Importing customers to database...")
        for i in range(0, len(customers_df), batch_size):
            batch = customers_df.iloc[i:i + batch_size]
            batch.to_sql('customers', engine, if_exists='append', index=False, method='multi')
            logger.info(f"Imported customers batch {i//batch_size + 1} of {(len(customers_df) // batch_size) + 1}")
        
        # Clean and import valid orders
        logger.info("Processing valid orders...")
        valid_customer_ids = set(customers_df['customer_id'].unique())
        valid_orders_df = clean_order_data(orders_df, valid_customer_ids)
        
        logger.info("Importing valid orders to database...")
        for i in range(0, len(valid_orders_df), batch_size):
            batch = valid_orders_df.iloc[i:i + batch_size]
            batch.to_sql('orders', engine, if_exists='append', index=False, method='multi')
            logger.info(f"Imported orders batch {i//batch_size + 1} of {(len(valid_orders_df) // batch_size) + 1}")
        
        # Final summary
        logger.info(f"""
            Import Summary:
            --------------
            Customers imported: {len(customers_df)}
            Valid orders imported: {len(valid_orders_df)}
            Orphaned orders (saved to CSV): {len(orders_df) - len(valid_orders_df)}
        """)

    except Error as e:
        logger.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during data import: {str(e)}")
        raise
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    # # For SQLAlchemy version
   
    
    # For mysql-connector version
    DB_CONFIG = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }

    CONNECTION_STRING = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    # Choose which version to run
    try:
        # SQLAlchemy version
        import_data_sqlalchemy(
            customer_file='data/customers.csv',
            order_file='data/order.csv',
            connection_string=CONNECTION_STRING,
            batch_size=1000
        )
        
        # # mysql-connector version
        # import_data_connector(
        #     customer_file='data/customers.csv',
        #     order_file='data/order.csv',
        #     db_config=DB_CONFIG,
        #     batch_size=1000
        # )
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")