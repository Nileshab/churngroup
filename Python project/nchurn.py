import streamlit as st
import numpy as np
import pandas as pd 
import seaborn as sns
import plotly.express as px 
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
from plotly.subplots import make_subplots
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Importing data
df1=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df=pd.read_csv('churn_data.csv')

# Designing page
st.set_page_config(page_title='CUSTOMER CHURN PREDICTION', layout='wide', page_icon="random")
st.title("CUSTOMER CHURN PREDICTION")
st.divider()
st.sidebar.divider()
selected_tab = st.sidebar.radio("OVERVIEW", ['Introduction','Dataset','EDA','Model','Conclusion','Prediction/Application'])
st.sidebar.divider()

if selected_tab == 'Introduction':
    st.markdown("In developed countries, telecommunications is a critical industry where customer attrition, or churn, poses a significant challenge, particularly in mature markets. Churn can be accidental, due to changes in a customer's circumstances, or intentional, when customers switch to competitors. Companies prioritize reducing voluntary churn, which results from controllable factors like billing or customer support. Retention is crucial for subscription-based models, as losing customers leads to economic and reputational harm. This study focuses on identifying the main causes of churn among fixed telephony subscribers in a telecom company.")
    c1,c2 = st.columns(2)
    with c1:
        st.image('chn.jpg')
    with c2:
        st.subheader('Business Understanding Churn Prediction')
        st.write('Identifying customers who are likely to cancel their contracts soon.')
        st.write('If the company can do that, it can handle users before churn.The target variable that we want to predict is categorical and has only two possible outcomes: churn or not churn (Binary Classification).We also would like to understand why the model thinks our customers churn, and for that, we need to be able to interpret the model’s predictions.')
    st.divider()
    

elif selected_tab == 'Dataset':
    st.subheader('Data overview')
    st.divider()
    st.subheader('Dataset')
    df1
    st.divider()
    st.subheader('According to the description, this dataset has the following information:')
    st.write('Services of the customers: phone; multiple lines; internet; tech support and extra services such as online security, backup, device protection, and TV streaming.')
    st.write('Account information: how long they have been clients, type of contract, type of payment method.')
    st.write('Charges: how much the client was charged in the past month and in total.')
    st.write('Demographic information: gender, age, and whether they have dependents or a partner')
    st.write('Churn: yes/no, whether the customer left the company within the past month.')
    st.divider()


elif selected_tab == 'EDA':
    st.subheader('Exploratory Data Analysis')
    st.divider()
    # Assuming df is already loaded as your DataFrame

    # 1. Distribution of Churn
    st.subheader("Distribution of Churn")
    churn_distribution = df['Churn'].value_counts().reset_index()
    churn_distribution.columns = ['Churn', 'Count']
    fig1 = px.bar(churn_distribution, x='Churn', y='Count', title='Churn Distribution')
    st.plotly_chart(fig1)

    # 2. Customer Demographics
    st.subheader("Customer Demographics and Churn")

    # Gender
    gender_churn = df.groupby(['gender', 'Churn']).size().reset_index(name='Count')
    fig2 = px.bar(gender_churn, x='gender', y='Count', color='Churn', barmode='group', title='Churn by Gender')
    st.plotly_chart(fig2)

    # Senior Citizen
    senior_churn = df.groupby(['SeniorCitizen', 'Churn']).size().reset_index(name='Count')
    fig3 = px.bar(senior_churn, x='SeniorCitizen', y='Count', color='Churn', barmode='group', title='Churn by Senior Citizen')
    st.plotly_chart(fig3)

    # Partner and Dependents
    partner_dependents_churn = df.groupby(['Partner', 'Dependents', 'Churn']).size().reset_index(name='Count')
    fig4 = px.bar(partner_dependents_churn, x='Partner', y='Count', color='Churn', facet_col='Dependents', barmode='group', title='Churn by Partner and Dependents')
    st.plotly_chart(fig4)

    # 3. Tenure Analysis
    st.subheader("Tenure Analysis and Churn")
    fig5 = px.histogram(df, x='tenure', color='Churn', nbins=50, title='Tenure Distribution by Churn')
    st.plotly_chart(fig5)

    # 4. Services Subscribed
    st.subheader("Services Subscribed and Churn")

    # Phone Service
    phone_churn = df.groupby(['PhoneService', 'Churn']).size().reset_index(name='Count')
    fig6 = px.bar(phone_churn, x='PhoneService', y='Count', color='Churn', barmode='group', title='Churn by Phone Service')
    st.plotly_chart(fig6)

    # Internet Service
    internet_churn = df.groupby(['InternetService', 'Churn']).size().reset_index(name='Count')
    fig7 = px.bar(internet_churn, x='InternetService', y='Count', color='Churn', barmode='group', title='Churn by Internet Service')
    st.plotly_chart(fig7)

    # Multiple Lines
    multiplelines_churn = df.groupby(['MultipleLines', 'Churn']).size().reset_index(name='Count')
    fig8 = px.bar(multiplelines_churn, x='MultipleLines', y='Count', color='Churn', barmode='group', title='Churn by Multiple Lines')
    st.plotly_chart(fig8)

    # 5. Contract Type
    st.subheader("Contract Type and Churn")
    contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
    fig9 = px.bar(contract_churn, x='Contract', y='Count', color='Churn', barmode='group', title='Churn by Contract Type')
    st.plotly_chart(fig9)

    # 6. Billing Preferences
    st.subheader("Billing Preferences and Churn")

    # Paperless Billing
    paperlessbilling_churn = df.groupby(['PaperlessBilling', 'Churn']).size().reset_index(name='Count')
    fig10 = px.bar(paperlessbilling_churn, x='PaperlessBilling', y='Count', color='Churn', barmode='group', title='Churn by Paperless Billing')
    st.plotly_chart(fig10)

    # Payment Method
    paymentmethod_churn = df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count')
    fig11 = px.bar(paymentmethod_churn, x='PaymentMethod', y='Count', color='Churn', barmode='group', title='Churn by Payment Method')
    st.plotly_chart(fig11)

    # 7. Charges Analysis
    st.subheader("Charges Analysis and Churn")

    # Monthly Charges
    fig12 = px.box(df, x='Churn', y='MonthlyCharges', title='Monthly Charges by Churn')
    st.plotly_chart(fig12)

    # Total Charges
    fig13 = px.box(df, x='Churn', y='TotalCharges', title='Total Charges by Churn')
    st.plotly_chart(fig13)

    # 8. Correlation Matrix
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot(plt)

    # 9. Customer Service Impact
    st.subheader("Customer Service Impact and Churn")

    # Tech Support
    techsupport_churn = df.groupby(['TechSupport', 'Churn']).size().reset_index(name='Count')
    fig14 = px.bar(techsupport_churn, x='TechSupport', y='Count', color='Churn', barmode='group', title='Churn by Tech Support')
    st.plotly_chart(fig14)

    # Online Security
    onlinesecurity_churn = df.groupby(['OnlineSecurity', 'Churn']).size().reset_index(name='Count')
    fig15 = px.bar(onlinesecurity_churn, x='OnlineSecurity', y='Count', color='Churn', barmode='group', title='Churn by Online Security')
    st.plotly_chart(fig15)

    # Device Protection
    deviceprotection_churn = df.groupby(['DeviceProtection', 'Churn']).size().reset_index(name='Count')
    fig16 = px.bar(deviceprotection_churn, x='DeviceProtection', y='Count', color='Churn', barmode='group', title='Churn by Device Protection')
    st.plotly_chart(fig16)

    # 10. Churn by Tenure and Monthly Charges
    st.subheader("Churn by Tenure and Monthly Charges")
    fig17 = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn', title='Churn by Tenure and Monthly Charges')
    st.plotly_chart(fig17)
    st.divider()

elif selected_tab == 'Model':
    st.subheader('Model Building')
    df1.columns = df.columns.str.lower()
    df1['totalcharges'] = df1['totalcharges'].replace(' ', 0)
    df['seniorcitizen'] = df1['seniorcitizen'].astype('object')
    df1['totalcharges'] = df1['totalcharges'].astype('float')
    df1.drop(columns=['customerid'], inplace=True)
    df1.drop_duplicates(inplace=True)
    num_cols = df1.select_dtypes(include='number').columns.tolist()
    cat_cols = df1.select_dtypes(include='object').columns.tolist()
    df1['tenure_group'] = pd.cut(df1.tenure, bins=3, labels=['low', 'medium', 'high'])
    df1['monthlycharges_group'] = pd.cut(df1.monthlycharges, bins=3, labels=['low', 'medium', 'high'])
    df1['totalcharges_group'] = pd.cut(df1.totalcharges, bins=3, labels=['low', 'medium', 'high'])
    # print(df.head())
    X = df1.drop(['churn', 'tenure', 'monthlycharges', 'totalcharges'], axis=1)
    y = df1['churn']
    print(X.columns)
    l = LabelEncoder()
    for i in X.columns:
    df1[i] = l.fit_transform(df1[i])
    X = df1.drop(['churn', 'tenure', 'monthlycharges', 'totalcharges'], axis=1)
    y = df1['churn']
    print(df1.head())
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)

    st = StandardScaler()
    x_train_0 = st.fit_transform(x_train)
    x_test_0 = st.transform(x_test)
    smote = SMOTE(sampling_strategy=0.8,k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(x_train_0, y_train)

    base_estimator = DecisionTreeClassifier(class_weight='balanced', max_depth=1)
    adb = AdaBoostClassifier(base_estimator,n_estimators=100)
    adb.fit(X_train_res, y_train_res)
    print(f"ADABOOST MODEL : {classification_report(y_test,adb.predict(x_test_0))})")

    from sklearn.svm import SVC

    svm = SVC(kernel='poly', C=100)
    svm.fit(X_train_res, y_train_res)

    print('Train score:', svm.score(X_train_res, y_train_res))
    print('Test score:', svm.score(x_test_0, y_test))
    print("SVM MODEL : f{classification_report(y_test,adb.predict(x_test_0)))")


    # ********************************************************************* (Above Code is Done for Converting into categroies)
    newdf = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    newdf.drop(columns=['customerID'], inplace=True)
    newdf.columns = newdf.columns.str.lower()
    newdf['totalcharges'] = newdf['totalcharges'].replace(' ', 0)
    newdf['seniorcitizen'] = newdf['seniorcitizen'].astype('object')
    newdf['totalcharges'] = newdf['totalcharges'].astype('float')
    newdf.drop_duplicates(inplace=True)
    num_cols = newdf.select_dtypes(include='number').columns.tolist()
    cat_cols = newdf.select_dtypes(include='object').columns.tolist()

    l = LabelEncoder()
    for i in cat_cols:
    newdf[i] = l.fit_transform(newdf[i])
    X = newdf.drop(['churn'], axis=1)
    y = newdf['churn']
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.22)
    st = StandardScaler()
    x_train_0 = st.fit_transform(x_train)
    x_test_0 = st.transform(x_test)
    smote = SMOTE(sampling_strategy=0.8,k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(x_train_0, y_train)
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(C=0.01)
    logreg.fit(x_train,y_train)

    # print('Train score:', logreg.score(x_train,y_train))
    # print('Test score:', logreg.score(x_test, y_test))
    print(classification_report(y_test,logreg.predict(x_test)))


    svm = SVC(kernel='poly', C=100)
    svm.fit(x_train, y_train)
    print(f"SVM MODEL : {classification_report(y_test,adb.predict(x_test))})")

    base_estimator = DecisionTreeClassifier(class_weight='balanced', max_depth=1)
    adb = AdaBoostClassifier(base_estimator,n_estimators=100)
    adb.fit(x_train, y_train)
    print(f"ADABOOST MODEL : {classification_report(y_test,adb.predict(x_test))})")
    st.divider()

elif selected_tab == 'Conclusion':
    st.subheader('Conclusion on Telco Customer Churn Based on Key Features')
    st.write('The analysis of Telco Customer Churn using the specified features has provided valuable insights into the factors influencing customer retention. Each feature plays a significant role in determining whether a customer is likely to continue their service or churn. Here’s a summary of the key findings and recommendations:')
    
    st.subheader('Important Features')
    st.subheader('1. Demographic Features:')
    st.markdown('* Gender, Senior Citizen, Partner, Dependents:')
    st.write('o	Insight: While gender shows minimal direct impact on churn, senior citizens and customers without partners or dependents exhibit higher churn rates.')
    st.write('o	Recommendation: Tailor services and offers to meet the specific needs of senior citizens and single individuals, such as more accessible customer support or loyalty programs aimed at these groups.')
    
    st.subheader('2. Service-Related Features:')
    st.markdown('* Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies:')
    st.write('o	Insight: Customers who subscribe to multiple services (e.g., internet, streaming, tech support) tend to have a lower churn rate, especially when additional services like online security and backup are included.')
    st.write('o	Recommendation: Promote bundling of services and provide discounts for comprehensive packages to enhance customer satisfaction and reduce the likelihood of churn.')
    
    st.subheader('3. Contractual Features:')
    st.markdown('* Contract, Paperless Billing, Payment Method:')
    st.write('o	Insight: Customers with month-to-month contracts are more prone to churn, especially those using paperless billing or certain payment methods like automatic credit card payments.')
    st.write('o	Recommendation: Encourage customers to switch to longer-term contracts with benefits such as reduced rates or additional perks. Also, ensure that payment options are seamless and reliable to avoid billing-related churn.')
    
    st.subheader('4. Financial Features:')
    st.markdown('* Monthly Charges, Total Charges:')
    st.write('o	Insight: Higher monthly charges correlate with higher churn, particularly among customers who feel they are not receiving proportional value.')
    st.write('o	Recommendation: Consider offering tiered pricing models that align costs more closely with service usage. Additionally, monitor total charges to identify potential billing issues that might lead to dissatisfaction.')
    
    st.subheader('5. Tenure:')
    st.write('•	Insight: Tenure is a critical factor, with customers in their initial months being more likely to churn. Longer-tenured customers are generally more loyal.')
    st.write('•	Recommendation: Implement retention strategies focused on new customers, such as personalized follow-ups and satisfaction checks during the first few months of service.')
    st.divider()

    st.subheader('Churn reduction Solution')
    st.subheader('1. Enhance Customer Experience:')
    st.write('•	Personalized Offers: Implement targeted marketing campaigns offering personalized discounts and promotions to high-risk customers identified through predictive modeling.')
    st.write('•	Improved Customer Support: Invest in training and tools for customer support teams to proactively address customer concerns, especially during contract renewals or after service issues.')

    st.subheader('2. Optimize Service Plans:')
    st.write('•	Flexible Contract Options: Introduce more flexible contract plans that cater to different customer needs, reducing the likelihood of churn due to dissatisfaction with contract terms.')
    st.write('•	Bundling Services: Encourage customers to bundle multiple services (e.g., internet, TV, phone) by offering discounts, as customers with bundled services are generally less likely to churn.')

    st.subheader('3. Loyalty Programs:')
    st.write('•	Reward Long-Term Customers: Implement loyalty programs that offer rewards and benefits to long-term customers, making them feel valued and reducing the likelihood of them switching providers.')
    st.write('•	Early Intervention: Use predictive analytics to identify customers likely to churn and intervene with retention offers before they make the decision to leave.')

    st.subheader('4. Continuous Monitoring and Feedback:')
    st.write('•	Regular Customer Feedback: Regularly gather and analyze customer feedback to understand evolving needs and expectations, allowing the company to adapt its services accordingly.')
    st.write('•	Churn Analytics: Continuously monitor churn metrics and refine predictive models to stay ahead of trends and implement timely interventions.')
    st.divider()

elif selected_tab == 'Prediction/Application':
    st.subheader('Business Understanding Churn Prediction')
    st.divider()
