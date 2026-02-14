# Forecasting Agricultural Commodity Prices using Machine learning Models
Agricultural price prediction is a tough problem in agri-economic systems because 
commodity prices are highly volatile. They are affected by seasonal cycles, weather 
changes, transportation costs, and shifts in supply and demand. Predicting commodity 
prices helps farmers, traders, and policymakers make better decisions and achieve price 
stability. 
This project presents a complete machine learning framework for forecasting 
agricultural commodity prices in Telangana market yards. The study uses a dataset with 
over 200,000 records from government open data sources. These records represent daily 
transactions of major crops across various markets. The process includes data cleaning, 
feature engineering, model evaluation, and the launch of an interactive Streamlit web 
application. 
Several models were trained and tested, including Linear Regression, Decision Tree 
Regressor, Random Forest, XGBoost, and CatBoost. XGBoost performed the best with R² 
= 0.93 and MAE ≈ ₹200, followed closely by CatBoost with R² = 0.91. These models 
effectively captured complex, nonlinear relationships between variables such as market 
yard, commodity type, arrival quantity, and date.  
The developed system offers a user-friendly web interface. Users can select a specific yard, 
commodity, and date to predict the expected modal price. This system can empower 
farmers and traders by providing valuable insights and improving price transparency in the 
agricultural markets of Telangana.
