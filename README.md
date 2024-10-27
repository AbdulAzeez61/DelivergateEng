#
# Delivergate Data Engineer Internship Technical Test
#

This repository contains the solution for the Delivergate Data Engineer Internship Technical Test. The project is divided into several parts covering data import, database setup, Streamlit dashboard creation and machine learning.

# Project Structure
#### Part 1: Data Import and Database Setup
#### Part 2: Streamlit Application with Data Filters and Visualizations
#### Part 3:  Machine Learning Model for Predicting Repeat Customers

# Pre requisites

Python 3.12.7

MySQL Server

# Setup Instructions
#### 1. Repo Cloning
   
     git clone https://github.com/AbdulAzeez61/DelivergateEng.git
   
     cd DelivergateEng

#### 2. Install dependancies
  
     pip install -r requirements.txt
   
#### 3. MySQL Database Setup

   Use the `tablesAndDB.sql` to create the database and the necessary tables for the project

#### 4. Data Import

Create an .env file with DB_HOST, DB_USER, DB_PASSWORD, DB_NAME variables

Run the `dataImport.py` script to extract, transform and load the data into the MySQL database.

# Running the Streamlit Application

Run
```
streamlit app/streamlit_app.py 
```
to run the application on the local browser.
#### App Features

#### Sidebar Filters:
- Date range filter for order_date
  
- Slider to filter customers by total amount spent
  
- Incrementer to filter customers by order count
  
- Main Dashboard:
  
- Data table with filtered results
  
- Bar chart of top 10 customers by revenue
  
- Line chart showing revenue over time

- Summary metrics (Total revenue, unique customers, order count)
  
# Machine Learning Model
  A logistic regression model is included to predict repeat customers based on total orders and revenue. This is trained when the app is run. And an input of order counts and total expenditures can be used to predict if a customer will be a repeat spender or not.
  
  
