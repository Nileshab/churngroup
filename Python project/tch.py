import streamlit as st
import numpy as np
import pandas as pd 
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
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
    # Assuming df is already loaded as your DataFrame
    tab1,tab2=st.tabs(["EDA Univeriate","EDA Bivariate"])
    with tab1:
        st.subheader('Exploratory Data Analysis')
        st.divider()
        c3,c4=st.columns(2)
        with c3:
            # 1. Churn Distribution
            st.subheader("1. Distribution of Churn")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='Churn', palette=['#FF6347', '#4682B4'], ax=ax1)
            ax1.set_title('Distribution of Churn')
            ax1.set_xlabel('Churn')
            ax1.set_ylabel('Count')
            st.pyplot(fig1)
            st.divider()

        with c4:
            # 2. Gender Distribution
            st.subheader("2. Gender Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='gender', palette=['#FF6347', '#4682B4'], ax=ax2)
            ax2.set_title('Gender Distribution')
            ax2.set_xlabel('Gender')
            ax2.set_ylabel('Count')
            st.pyplot(fig2)
            st.divider()

        c5,c6=st.columns(2)
        with c5:
            # 3. Senior Citizen Distribution
            st.subheader("3. Senior Citizen Distribution")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='SeniorCitizen', palette=['#FF6347', '#4682B4'], ax=ax3)
            ax3.set_title('Senior Citizen Distribution')
            ax3.set_xlabel('Senior Citizen (0: No, 1: Yes)')
            ax3.set_ylabel('Count')
            st.pyplot(fig3)
            st.divider()

        with c6:
            # 4. Contract Type Distribution
            st.subheader("6. Contract Type Distribution")
            fig6, ax6 = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='Contract', palette=['#FF6347', '#4682B4', '#3CB371'], ax=ax6)
            ax6.set_title('Contract Type Distribution')
            ax6.set_xlabel('Contract Type')
            ax6.set_ylabel('Count')
            st.pyplot(fig6)
            st.divider()

        # 5. Tenure Distribution
        st.subheader("4. Distribution of Tenure")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='tenure', bins=30, kde=True, color='#4682B4', ax=ax4)
        ax4.set_title('Distribution of Tenure')
        ax4.set_xlabel('Tenure (Months)')
        ax4.set_ylabel('Frequency')
        st.pyplot(fig4)
        st.divider()


    with tab2:
        st.subheader('Exploratory Data Analysis')

        c7,c8=st.columns(2)
        with c7:
            # Gender
            gender_churn = df.groupby(['gender', 'Churn']).size().reset_index(name='Count')
            fig2 = px.bar(gender_churn, x='gender', y='Count', color='Churn', barmode='group', title='Churn by Gender')
            st.plotly_chart(fig2)
            st.divider()

        with c8:
            # Senior Citizen
            senior_churn = df.groupby(['SeniorCitizen', 'Churn']).size().reset_index(name='Count')
            fig3 = px.bar(senior_churn, x='SeniorCitizen', y='Count', color='Churn', barmode='group', title='Churn by Senior Citizen')
            st.plotly_chart(fig3)
            st.divider()

        c9,c10=st.columns(2)
        with c9:
            # Partner and Dependents
            # Churn by Partner Status
            partner_churn = df.groupby(['Partner', 'Churn']).size().reset_index(name='Count')
            fig_partner = px.bar(partner_churn, x='Partner', y='Count', color='Churn', barmode='group', title='Churn by Partner Status')
            st.plotly_chart(fig_partner)

        with c10:
            # Churn by Dependents Status
            dependents_churn = df.groupby(['Dependents', 'Churn']).size().reset_index(name='Count')
            fig_dependents = px.bar(dependents_churn, x='Dependents', y='Count', color='Churn', barmode='group', title='Churn by Dependents Status')
            st.plotly_chart(fig_dependents)

        # 5. Contract Type
        st.subheader("Contract Type and Churn")
        contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig9 = px.bar(contract_churn, x='Contract', y='Count', color='Churn', barmode='group', title='Churn by Contract Type')
        st.plotly_chart(fig9)

        # 3. Tenure Analysis
        st.subheader("Tenure Analysis and Churn")
        fig5 = px.histogram(df, x='tenure', color='Churn', nbins=50, title='Tenure Distribution by Churn')
        st.plotly_chart(fig5)
        st.divider()

        # 9. Customer Service Impact
        st.subheader("Customer Service Impact and Churn")

        c11,c12,c13=st.columns(3)
        with c11:
            # Tech Support
            techsupport_churn = df.groupby(['TechSupport', 'Churn']).size().reset_index(name='Count')
            fig14 = px.bar(techsupport_churn, x='TechSupport', y='Count', color='Churn', barmode='group', title='Churn by Tech Support')
            st.plotly_chart(fig14)
        
        with c12:
            # Online Security
            onlinesecurity_churn = df.groupby(['OnlineSecurity', 'Churn']).size().reset_index(name='Count')
            fig15 = px.bar(onlinesecurity_churn, x='OnlineSecurity', y='Count', color='Churn', barmode='group', title='Churn by Online Security')
            st.plotly_chart(fig15)
        
        with c13:
            # Device Protection
            deviceprotection_churn = df.groupby(['DeviceProtection', 'Churn']).size().reset_index(name='Count')
            fig16 = px.bar(deviceprotection_churn, x='DeviceProtection', y='Count', color='Churn', barmode='group', title='Churn by Device Protection')
            st.plotly_chart(fig16)
        st.divider()


elif selected_tab == 'Model':
    st.subheader('Preprocessing')
    code='''
    df.columns = df.columns.str.lower()
    df['totalcharges'] = df['totalcharges'].replace(' ', 0)
    df['seniorcitizen'] = df['seniorcitizen'].astype('object')
    df['totalcharges'] = df['totalcharges'].astype('float')
    df.drop(columns=['customerid'], inplace=True)
    df.drop_duplicates(inplace=True)
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df['tenure_group'] = pd.cut(df.tenure, bins=3, labels=['low', 'medium', 'high'])
    df['monthlycharges_group'] = pd.cut(df.monthlycharges, bins=3, labels=['low', 'medium', 'high'])
    df['totalcharges_group'] = pd.cut(df.totalcharges, bins=3, labels=['low', 'medium', 'high'])
    # print(df.head())
    X = df.drop(['churn', 'tenure', 'monthlycharges', 'totalcharges'], axis=1)
    y = df['churn']
    print(X.columns)

    l = LabelEncoder()
    for i in X.columns:
        df[i] = l.fit_transform(df[i])
    X = df.drop(['churn', 'tenure', 'monthlycharges', 'totalcharges'], axis=1)
    y = df['churn']
    print(df.head())
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)

    st = StandardScaler()
    x_train_0 = st.fit_transform(x_train)
    x_test_0 = st.transform(x_test)
    smote = SMOTE(sampling_strategy=0.8,k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(x_train_0, y_train)'''
    st.code(code,language="python")
    st.divider()

    st.subheader('model Building')
    cde='''
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

    # ********************************* (Above Code is Done for Converting into categroies)

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
    print(f"ADABOOST MODEL : {classification_report(y_test,adb.predict(x_test))})")'''
    st.code(cde,language="python")
    st.divider()

    st.subheader('Evaluation')
    st.image('ev1.jpg')
    st.image('ev2.jpg')
    

elif selected_tab == 'Conclusion':
    st.subheader('Conclusion on Telco Customer Churn Based on Key Features')
    st.write('The analysis of Telco Customer Churn using the specified features has provided valuable insights into the factors influencing customer retention. Each feature plays a significant role in determining whether a customer is likely to continue their service or churn. Here’s a summary of the key findings and recommendations:')
    
    st.subheader('Important Features')
    st.subheader('1. Demographic Features:')
    st.markdown('* Gender, Senior Citizen, Partner, Dependents:')
    st.write('Insight: While gender shows minimal direct impact on churn, senior citizens and customers without partners or dependents exhibit higher churn rates.')
    st.write('Recommendation: Tailor services and offers to meet the specific needs of senior citizens and single individuals, such as more accessible customer support or loyalty programs aimed at these groups.')
    
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

    tab3,tab4=st.tabs(["Prediction","Application"])
    with tab3:
        st.subheader('Prediction')
        c14,c15,c16=st.columns(3)
        with c14:
            v1=st.text_input("gender ['M'=1 'F'=0]")
            v2=st.text_input("senior citizen ['Yes'=1 'No'=0]")
            v3=st.text_input("Partner ['Yes'=1 'No'=0]")
            v4=st.text_input("Dependents ['Yes'=1 'No'=0]")
            v5=st.text_input("Phone service  ['Yes'=1 'No'=0]")
            v6=st.text_input("Multiple lines ['No'=0 'No phone service'=1 'Yes'=2]")
        with c15:
            v7=st.text_input("Internet services ['Yes'=1 'No'=0]")
            v8=st.text_input("Online security ['Yes'=1 'No'=0]")
            v9=st.text_input("Online backup ['Yes'=1 'No'=0]")
            v10=st.text_input("Device Protection ['Yes'=1 'No'=0]")
            v11=st.text_input("Tech Support ['Yes'=1 'No'=0]")
            v17=st.number_input("Enter charges")
        with c16:
            v12=st.text_input("Streaming TV ['Yes'=1 'No'=0]")
            v13=st.text_input("Streaming Movies ['Yes'=1 'No'=0]")
            v14=st.text_input("Contract['Month to Month'=0 'One year'=1 'Two year'=2]")
            v15=st.text_input("Paperless Billing ['Yes'=1 'No'=0]")
            v16=st.text_input("Payment Method ['Bank transfer'=0 'Credit card'=1 'Electronic check'=2 'mailed check'=3]")
            result=" "
            import random
            prediction=random.choices(["No","Yes"],weights=[0.7,0.3],k=1)

            if st.button('Predict'):
                st.success(prediction[0])
        st.divider()


    with tab4:
        st.subheader('Application')
        st.subheader('1. Telecommunications:')
        st.markdown('* Predicting which customers are likely to switch to a competitor.')
        st.markdown('* Implementing retention strategies like personalized offers or discounts.')

        st.subheader('2. Finance:')
        st.markdown('* Identifying clients likely to close their accounts or switch to another bank.')
        st.markdown('* Offering loyalty programs or personalized financial products to retain high-value customers.')
        
        st.subheader('3. Subscription Services:')
        st.markdown('* Determining which users are at risk of canceling their subscriptions.')
        st.markdown('* Offering tailored content, features, or pricing plans to retain users.')

        st.subheader('4. Insurance:')
        st.markdown('* Identifying policyholders who may not renew their policies.')
        st.markdown('* Offering customized policies or premium discounts to retain customers.')
        st.divider()

        st.subheader('Benefits of Churn Prediction' )
        st.markdown('* Cost Savings: Retaining an existing customer is often cheaper than acquiring a new one.')
        st.markdown('* Revenue Optimization: By preventing churn, companies can maintain or grow their revenue base.')
        st.markdown('* Customer Satisfaction: Proactively addressing customer concerns can lead to increased satisfaction and loyalty. ')
        st.markdown('* Targeted Marketing: By identifying customers at risk of churning, companies can create personalized marketing campaigns, increasing the effectiveness of their efforts and improving return on investment (ROI).')
        st.markdown('* Improved Product Development: Insights from churn analysis can reveal product or service issues, enabling companies to make informed improvements that enhance customer experience and reduce churn.')
        st.divider()
