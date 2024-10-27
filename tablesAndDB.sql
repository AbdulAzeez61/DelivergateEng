-- Create database
CREATE DATABASE IF NOT EXISTS delivergate_db_3;
USE delivergate_db_3;

-- Create customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(255)
);

-- Create orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(10) NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    order_date DATETIME NOT NULL,
    customer_id INT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    UNIQUE KEY unique_order_id (order_id)
);
