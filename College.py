import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from scipy import stats

st.set_page_config(layout = 'wide')

original = pd.read_csv('college.csv')
numv = ['parent_age','parent_salary','house_area','average_grades']
catv = ['type_school', 'school_accreditation','gender','residence','parent_was_in_college','interest']
not_in_college = original.loc[original['in_college']==False]
in_college = original[original['in_college']==True]
numv_dict= {'parent_age':'Parent Age','parent_salary':'Parent Salary','house_area':'House Area','average_grades':'Average Grades'}
catv_dict = {'type_school':'Type School', 'school_accreditation':'School Accreditation','gender':'Gender','residence':'Residence','parent_was_in_college':'Parent Was In College','interest':'Interest'}
allv_dict = numv_dict.copy()
allv_dict.update(catv_dict)

def naive_bayes(df,target):
    num_vars = [col for col in df.columns if col in numv_dict.keys()]
    if len(num_vars) == 0:
        num_vars = None
    def fit_distribution(data):
        mu = np.mean(data)
        sigma = np.std(data)
        dist = stats.norm(mu, sigma)    
        return dist
    prior= dict(df[target].value_counts(normalize = True))
    target_true = df[df[target]==True].drop(target,axis = 1).copy()
    target_false = df[df[target]==False].drop(target,axis = 1).copy()
    if num_vars is not None:    
        num_target_true = {}
        num_target_false = {}
        for var in num_vars:
            num_target_true[var] = fit_distribution(target_true[var]).pdf(df[var])
            num_target_false[var] = fit_distribution(target_false[var]).pdf(df[var])
        num_target_true = pd.DataFrame(num_target_true)
        num_target_false = pd.DataFrame(num_target_false)
    else:
        num_vars = []
        num_target_true = pd.DataFrame()
        num_target_false = pd.DataFrame()
    cat_vars = [var for var in target_true.columns if var not in num_vars]
    if len(cat_vars) > 0:
        cat_target_true = {}
        cat_target_false = {}
        for var in cat_vars:
            cat_target_true[var] = df[var].map(dict(target_true[var].value_counts(normalize = True)))
            cat_target_false[var] = df[var].map(dict(target_false[var].value_counts(normalize = True)))
        cat_target_true = pd.DataFrame(cat_target_true)
        cat_target_false = pd.DataFrame(cat_target_false)
    else:
        cat_target_true = pd.DataFrame()
        cat_target_false = pd.DataFrame()
#compare target true and false, turn into series
    target_true_prob = pd.concat([num_target_true, pd.DataFrame(cat_target_true)], axis = 1).prod(axis = 1)
    target_false_prob = pd.concat([num_target_false, pd.DataFrame(cat_target_false)], axis = 1).prod(axis = 1)
    prediction = target_true_prob > target_false_prob
    accuracy = (prediction==df[target]).mean()
    return pd.DataFrame(zip(df[target],prediction), columns = ['Actual','Predicted']),accuracy

#sections

with st.sidebar:
        selected = option_menu(menu_title = 'Navigation Pane',
        options = ['Abstract','Background Information','Data Cleaning','Exploratory Analysis','Naive Bayes','Data Analysis','Conclusion'],
        menu_icon = 'arrow-down-right-circle-fill',
        icons = ['bookmark-check', 'book', 'box', 'map', 'circle','boxes', 'bar-chart'])

if selected == 'Abstract':
    st.title('College Case Study Abstract')
    st.markdown("The dataset this case study is based on consists of synthetic data collected on 1000 students, with information on 11 variables, including personal background (parent salary, gender, residence...etc.) and academic background (grades, school accreditation, school type, interest in going to college). The target variable for this dataset is the last column - 'in_college', represented in the type of boolean, where 'true' is in college and 'false' is not in college.")
    st.markdown('With these data, we will use several different visualizations and prediction models to explore and determine the correlations between each variable and the target variable - "in_college". This will help us draw conclusions on which variables matter the most, which matter the least, and what are the variables we truly need to consider in our prediction model. Then we will use the Naive Bayes algorithm to predict whether or not a certain student will go to college and test the model\'s accuracy.')
    

if selected == 'Background Information':
    st.title('Background Information')
    st.markdown('College. The thing that "determines your future". ')
    st.markdown("From a young age, many kids are overwhelmed by their parents with all kinds of classes, summer camps, extracurriculars...etc. hoping to construct the perfect résumé and path for their kids and maximize their chance of getting in into a \"good college\". Nowadays, more and more parents  But why is college so important to people? College is often, not always correctly, percieved as the benchmark and criteria for determining a person's level of education, how successful they will be, and the way to get a lucrative career. It might also have different implications for families in different situations. For example, a family with a record of people that went to college before might set higher and higher goals/requirements for their kids, such as going to a \"better college\" than the college they went to. On the other hand, for families that are struggling financially and without a previous college record might see sending their kids to college as the only way to relese their burden and gain a brighter, better future. However, the amount of people hoping to go to college far exceeds the amount of people that can actually make that dream come true. With that said, we might as well try to find the answer to the ultimate question: can I get into college?")
    st.markdown("")
    st.markdown("There are many, or prossibly an infinite amount of factors that could influence one's chances of getting into college, including their background (parent salary, gender, residence...etc.) and their academic background (grades, school accreditation, school type, interest in going to college). There is no perfect solution to getting admitted to college; at the end of the day, college admission officers are humans and they make the decision based on what they see in a student as a person. But there are certain \"requirements\" that many colleges will require, such as GPA (grades) or if the family is able to pay for tuition, and using these academic and personal background information, we can try to predict a student's chance of going to college.")
    st.markdown('Here are some links to learn more about this topic:')
    st.markdown('https://prepory.com/blog/what-do-college-admissions-officers-look-for-in-an-applicant/#:~:text=Basically%2C%20there%20are%20six%20main,recommendation%2C%20and%20your%20personal%20statement.')
    st.markdown("https://www.snhu.edu/about-us/newsroom/education/why-is-college-important#:~:text=College%20is%20important%20for%20many,an%20impact%20on%20your%20community.&text=With%20more%20and%20more%20careers,your%20success%20in%20today's%20workforce.")
    st.markdown("https://www.collegedata.com/resources/getting-in/what-do-colleges-look-for-in-students")

if selected == 'Data Cleaning':
    st.title('Data Cleaning')
    st.markdown('For this case study, the dataset is already synthesized data that doesn\'t require much cleaning.')
    st.write(original)
    st.caption('https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset')
    st.subheader('Introduction to each variable:')
    st.markdown('Type School - what type of school the student attends (Vocational/Academic)')
    st.markdown('School Accreditation - school accreditation of school the student went to (A/B)')
    st.markdown('Gender - gender of student (Male/Female)')
    st.markdown('Interest - student\'s interest in college (Not interested/Less interested/Quiet interested/Uncertain/Very interested)')
    st.markdown('Residence - student\'s residence (Urban/Rural)')
    st.markdown('Parent Age - age of student\'s parent (numeric)')
    st.markdown('Parent Salary - parent salary (numeric)')
    st.markdown('House Area - house area (numeric)')
    st.markdown('Average Grades - student\'s average grades (numeric)')
    st.markdown('Parent was in college - whether or not student\'s parent was in college (True/False)')
    st.markdown('In College - did the student go to college (True/False)')

if selected == 'Exploratory Analysis':
    st.title('Exploratory Analysis')
    st.markdown("To find the correlation between variables that contribute to getting into college in a straightfoward manner, there are several ways to do so by visualizing data using maps, graphs, plots...etc. In this page, you will be able to test out these visualizations and try different combinations of data/variables using the options given beside each graph.")
   
    #Box plot for numeric variables in terms of in college
    st.header('Numeric variables vs "In College" variable (box plots)')
    col1,col2 = st.columns([3,5])
    numv_choice_box_pretty = col1.selectbox('Select a variable',numv_dict.values())
    numv_choice_box = [k for k,v in numv_dict.items() if v == numv_choice_box_pretty ]
    def box_plot_num (var):
        trace1 = go.Box(x = in_college[var], name = 'In College')
        trace2 = go.Box(x = not_in_college[var], name = 'Not In College')
        box = go.Figure(data=[trace1, trace2])
        title = var.replace('_',' ').title()
        box.update_layout(
            title= f'<b>{title} vs In College <b>',
            xaxis_title=title,
            legend_title="Legend")
        return box
    num_box = box_plot_num(numv_choice_box[0])
    col2.plotly_chart(num_box)
    col1.markdown('These box plots allows us to observe the difference of spread between student is in college and not in college for different variables. It gives us an initial and general idea as to weather or not the variable is related to getting into college by looking at the extent to which the two box plots are different/the same.')
    col1.markdown('All of the variables show a noticable difference between the two box plots, indicating a somewhat dependent relationship between these variables and the target variable - "in college"')
    col1.markdown('Similarily, we can use histograms to demonstrate this.')

    st.markdown('')
    #Histogram for numeric variables in terms of in college
    st.header('Numeric variables vs "In College" variable (histograms)')
    col3,col4 = st.columns([3,5])
    numv_choice_hist_pretty = col3.selectbox('Select a variable',numv_dict.values(),key = '<what>')
    numv_choice_hist = [k for k,v in numv_dict.items() if v == numv_choice_hist_pretty]
    def hist_num (var):
        trace1 = go.Histogram(x = in_college[var], name = 'In College')
        trace2 = go.Histogram(x = not_in_college[var], name = 'Not In College')
        hist = go.Figure(data = [trace1,trace2])
        title = var.replace('_',' ').title()
        hist.update_layout(
            title = f'<b>{title} vs In College <b>',
            xaxis_title = title,
            legend_title = 'Legend',
            barmode = 'overlay')
        hist.update_traces(opacity = 0.5, 
        marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
        return hist
    num_hist = hist_num(numv_choice_hist[0])
    col4.plotly_chart(num_hist)
    col3.markdown('These histograms seem to support the observations from the box plots above. The overlayed histograms seem to have noticable differences in terms of spread. For example, average grades seem to be higher in general for students who went to college compared to those who did not.')
    col3.markdown('This supports the hypothesis that these numeric variables are correlated to the "in college" variable. Now lets look at the categorical variables.')

    #Bar charts for categorical variables in terms of in college
    st.header('Categorical variables vs "In College" variable (Bar charts)')
    col5,col6 = st.columns([3,5])
    catv_choice_hist_pretty = col5.selectbox('Select a variable',catv_dict.values(),key='<something>')
    catv_choice_hist = [k for k,v in catv_dict.items() if v == catv_choice_hist_pretty]
    def hist_cat(var):
        title = var.replace('_', ' ').title()
        hist = px.histogram(original,x=var, color = 'in_college',barmode = 'group',title = f'<b>{title} vs In College <b>',
        labels = {'in_college':'Legend','True':'In College','False':'Not In College',var:title})
        hist.update_layout()
        return hist
    cat_hist = hist_cat(catv_choice_hist[0])
    col6.plotly_chart(cat_hist)
    col5.markdown('These histograms allow us to observe the distribution of categorical variables in terms of "in college", which indicates to some extent how correlated the variables are. The bigger the difference between each bar of the same category, the more correlated the variables are.')
    col5.markdown('All the variables seem pretty evenly distributed, though there are still minor differences. Two of the most correlated variable we can observe from these graphs is "parent was in college" and "Interest". If we look closely, we can realize that these differences are somehow reasonable and explainable. For example, for "parent was in college", for the students whos parents didnt go to college, more of themm did not go to college compared to the ones that did, and if parents did go to college, it was the opposite.')


    #Numeric comparison scatter plot
    st.header('Numerical variables vs Numerical variable vs "In College" (scatter plot)')
    col7,col8 = st.columns([3,5])
    def compare(xvar,yvar):
        comp = px.scatter(original, x=xvar, y=yvar, color="in_college", marginal_y="box",
           marginal_x="box", trendline="ols", template="presentation")
        xvar_title = xvar.replace('_', ' ').title()
        yvar_title = yvar.replace('_', ' ').title()
        comp.update_layout(title_text = f'<b>{xvar_title} vs {yvar_title}</b>') 
        return comp
    with st.form('Select 2 variables'):
        xvar_pretty = col7.selectbox('Select a x-axis variable',numv_dict.values(),key = "<random>")
        yvar_pretty = col7.selectbox('Select a y-axis variable',numv_dict.values(),key = "<random>")
        xvar = [k for k,v in numv_dict.items() if v == xvar_pretty]
        yvar = [k for k,v in numv_dict.items() if v == yvar_pretty]
        num_compare = compare(xvar[0],yvar[0])
        submitted=st.form_submit_button("Submit to compare the two variables")
        if submitted:
            col8.plotly_chart(num_compare)
    col7.markdown('This scatter plot consists of three variables represented on the x-axis, y-axis, and color. This allows us to compare two variables or observe two variables together compared to the "in_college" variable to find correlations between these variables. Furthermore, there are box plots on top and on the right side of the graph, once again giving us an idea of the distribution of data for each variable.')
    col7.markdown('We can find some interesting trends through these graphs, for example, when comparing economic variables such as house_area and parent_salary, we can observe that generally students from financially better families are more likely to go to college. We can also see that even though some variables may seem like they might have some correlation with eachother, such as parent salary and house area, the numeric variables seem to have very weak correlations with eachother, which is shown through the trendline.')


    #Categorical comparison histogram
    st.header('Categorical variable vs Categorical variable (Histogram)')
    col9,col10 = st.columns([3,5])
    def compare_cat(xvar,color,nbins = None):
        fig = px.histogram(original, x = xvar, color = color,barmode = 'group',title = xvar.replace('_',' ').title(),
                      histnorm = 'percent',nbins = nbins)
        return fig
    with st.form('plzSelect 2 variables'):
        xvar_pretty = col9.selectbox('Select a x-axis variable',catv_dict.values(),key ='whatever')
        color_pretty = col9.selectbox('Select a color variable',catv_dict.values(),key='wh')
        xvar = [k for k,v in catv_dict.items() if v== xvar_pretty]
        color = [k for k,v in catv_dict.items() if v== color_pretty]
        cat_compare = compare_cat(xvar[0],color[0])
        submitted = st.form_submit_button('Submit to compare the two variables')
        if submitted:
            col10.plotly_chart(cat_compare)
    col9.markdown('These histograms utilize color and bins to compare betweeen different categorical variables. These graphs do not concern the variable "in_college"; it focuses on the correlation between different categorical variables. In these graphs, the comparison between the bars of different colors of the same category can help us determine correlation between categorical variables.')
    col9.markdown('After exploring with these graphs, we can see that there are some variables that have pretty even bars for each category, indicating a weak correlation, while some have notable differences that indicate strong correlations. For example, type school and gender doesnt have a strong correlation, while parent was in college and type school has a relatively strong correlation.')


if selected == 'Naive Bayes':

    st.title('Naive Bayes Model for college prediction')
    st.markdown("The Naive Bayes algorithm is a classification technique based on Bayes' Theorem with an independence assumption among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. The Bayes' Theorem provides a way that we can calculate the probability of a piece of data belonging to a given class, given our prior knowledge.")
    st.markdown('In this section, you will be able to experiment with the naive bayes model used to make predictions on college acceptance.')
    st.caption('It is suggested that you visit the Data Analysis section of this case study before proceeding the following section. It will give you an idea of what this section is and what you are trying to do here.')
    
    col11,col12 = st.columns([3,5])
    

    with st.form('hi'):
        options = col11.multiselect('Select the variables you want to include in the model (multiple choice)',catv+numv,key = 'keys')
        submitted = st.form_submit_button('Submit to see the result')
        if submitted:
            naive_bayes_result = naive_bayes(original[['in_college']+options],'in_college')
            col12.write(f'The accuracy of Naive Bayes: {naive_bayes_result[1]}')
            col12.write(naive_bayes_result[0])

    st.caption('The following is a block of code that calculated the naive bayes result; if you are interested, you can expand the section to get a look behind the scenes.')
    with st.expander('Click to read the naive bayes function'):
        body = '''#define the function
def naive_bayes(df,target):
    #identify numeric and categorical variables of df
    num_vars = [col for col in df.columns if col in numv_dict.keys()]
    if len(num_vars) == 0:
        num_vars = None
    #helper function that makes a normal distribution from numeric variables
    def fit_distribution(data):
        mu = np.mean(data)
        sigma = np.std(data)
        dist = stats.norm(mu, sigma)    
        return dist
    #prior probability of target variable
    prior= dict(df[target].value_counts(normalize = True))
    #create two dataframes, target variable true and target variable false
    target_true = df[df[target]==True].drop(target,axis = 1).copy()
    target_false = df[df[target]==False].drop(target,axis = 1).copy()
    #numerical variables in each subset are fitted with normal distributions
    if num_vars is not None:    
        num_target_true = {}
        num_target_false = {}
        for var in num_vars:
            num_target_true[var] = fit_distribution(target_true[var]).pdf(df[var])
            num_target_false[var] = fit_distribution(target_false[var]).pdf(df[var])
        num_target_true = pd.DataFrame(num_target_true)
        num_target_false = pd.DataFrame(num_target_false)
    else:
        num_vars = []
        num_target_true = pd.DataFrame()
        num_target_false = pd.DataFrame()
    #categorical variables in each subset are converted into probabilities based on their frequency in each subset
    cat_vars = [var for var in target_true.columns if var not in num_vars]
    if len(cat_vars) > 0:
        cat_target_true = {}
        cat_target_false = {}
        for var in cat_vars:
            cat_target_true[var] = df[var].map(dict(target_true[var].value_counts(normalize = True)))
            cat_target_false[var] = df[var].map(dict(target_false[var].value_counts(normalize = True)))
        cat_target_true = pd.DataFrame(cat_target_true)
        cat_target_false = pd.DataFrame(cat_target_false)
    else:
        cat_target_true = pd.DataFrame()
        cat_target_false = pd.DataFrame()
    #compare target true and false, turn into series
    target_true_prob = pd.concat([num_target_true, pd.DataFrame(cat_target_true)], axis = 1).prod(axis = 1)
    target_false_prob = pd.concat([num_target_false, pd.DataFrame(cat_target_false)], axis = 1).prod(axis = 1)
    prediction = target_true_prob > target_false_prob
    accuracy = (prediction==df[target]).mean()
    #return dataframe with actual and predicted values of the target variable
    return pd.DataFrame(zip(df[target],prediction), columns = ['Actual','Predicted']),accuracy
    '''
        st.code(body,language = 'python')
    

if selected == 'Data Analysis':
    st.title('Data Analysis')
    st.header('Naive Bayes Prediction Model')

    col13,col14 = st.columns([1,1])
    col13.markdown("First, I didn't leave out any variables and used all of them in the test. The result had an 83% accuracy, which was pretty good since the data was evenly split among in college and not in college students.")
    naive_bayes_result1 = naive_bayes(original,'in_college')
    col14.markdown('Naive Bayes result of full dataset')
    col14.write(f'The accuracy of Naive Bayes: {naive_bayes_result1[1]}')
    col14.write(naive_bayes_result1[0])
    col13.markdown('To improve the accuracy of the naive bayes test, we can experiment with different sets of variables and see how they affect the accuracy. Theoretically, if we include only the most correlated variables to the target variable, we would have a higher accuracy. Hence, the first step is to find out which are the most correlated and least correlated variables to "in_college" variable.')
    col13.markdown('From the graphs and charts in the exploratory analysis section, we have a basic idea of the correlation between different numeric and categorical variables to the target variable, but to be more clear, we can first use the chi squre test on the categorical variables of the dataset.')

    st.subheader('Chi Square Test')
    col15,col16 = st.columns([3,5])
    col15.markdown('Chi Square Test is a hypothesis test that compares the expected values of a dateset to the observed/actual values of the dataset to determine whether or not/to what extent are those variables independent of eachother. The test yeilds two results - the chi square value and the p-value.')
    col15.markdown('On the right is a dataframe containing the results of the chi square test of each variable, including the chi square value and the p-value.')
    chi2 = pd.DataFrame(columns = ['Variable','chi squared','p value'])
    for column in catv:
        table = pd.crosstab(original[column],original['in_college'])
        row = pd.DataFrame({'Variable':[column],'chi squared':[stats.chi2_contingency(table)[0]],'p value':[stats.chi2_contingency(table)[1]]})
        chi2 = pd.concat([chi2,row])
    chi2['Variable'] = chi2['Variable'].replace(catv_dict)
    col16.dataframe(chi2.sort_values('p value').reset_index(drop = True))
    col15.markdown('')
    col15.markdown('We can visualize the results by putting them into a bar graph -->')
    chi_bar=px.bar(chi2,x='chi squared',y='Variable',orientation='h')
    col16.write(chi_bar)
    col15.markdown('From the bar graph, we can identify that some of the most correlated variables are interest and parent was in college, while school accreditation and residence are less correlated.')
    col15.markdown('Using this, we can run the naive bayes prediction model again after dropping some of the less correlated variables to hopefully increase the accuracy of the prediction.')

    st.subheader('Adjustment 1')
    col17,col18 = st.columns([1,1])
    most_correlated = original.drop(['school_accreditation','residence','gender','type_school'],axis = 1)
    naive_bayes_result2 = naive_bayes(most_correlated,'in_college')
    col18.write(f'The accuracy of Naive Bayes: {naive_bayes_result2[1]}')
    col18.write(naive_bayes_result2[0])
    col17.markdown('On the right side is the result of the Naive Bayes test on the original dataframe after dropping some less correlated variables, including school accreditation, residence, gender, and type school.')
    col17.markdown('As we can see, the accuracy of the Naive bayes test did not improve, but instead, it decreased by 0.02 percent, which means we should keep those variables in the dataset afterall.')
    col17.markdown('Now, we will look at numeric variables and their correlation with the target variable to make another set of adjustments to the dataset. Using the box plots shown down below, we can tell that all these numeric variables seem to be pretty correlated to the "in_college" variable except for parent age, which was highly uncorrelated.')

    facets = make_subplots(rows = 2,cols=2, subplot_titles = ('Parent age vs In college','Average grades vs In college','House area vs In college','Parent salary vs In college'),horizontal_spacing = 0.2,vertical_spacing = 0.3)
    facets.add_trace(go.Box(x = in_college['parent_age'], name = 'In College',showlegend=False),row=1,col=1)
    facets.add_trace(go.Box(x = not_in_college['parent_age'], name = 'Not In College',showlegend=False),row=1,col=1)
    facets.add_trace(go.Box(x = in_college['average_grades'], name = 'In College',showlegend=False),row=1,col=2)
    facets.add_trace(go.Box(x = not_in_college['average_grades'], name = 'Not In College',showlegend=False),row=1,col=2)
    facets.add_trace(go.Box(x = in_college['house_area'], name = 'In College',showlegend=False),row=2,col=1)
    facets.add_trace(go.Box(x = not_in_college['house_area'], name = 'Not In College',showlegend=False),row=2,col=1)
    facets.add_trace(go.Box(x = in_college['parent_salary'], name = 'In College',showlegend=False),row=2,col=2)
    facets.add_trace(go.Box(x = not_in_college['parent_salary'], name = 'Not In College',showlegend=False),row=2,col=2)
    facets.update_xaxes(title='Parent Age', row=1, col=1)
    facets.update_xaxes(title='Average Grades', row=1, col=2)
    facets.update_xaxes(title='House Area', row=2, col=1)
    facets.update_xaxes(title='Parent Salary', row=2, col=2)
    facets.update_layout(title_text = '<b>Numeric variables vs In college<b>')
    st.plotly_chart(facets)

    st.subheader('Adjustment 2')
    col19,col20 = st.columns([1,1])
    col19.markdown('Because parent age seemed to be obviously uncorrelated, we will drop the variable and run the Naive Bayes test again.')
    col19.markdown('The results are shown on the right. As we can see, there is an decrease in accuracy again, meaning we should still keep the parent age variable in the dataset.')
    more_correlated = original.drop(['parent_age'],axis = 1)
    naive_bayes_result3 = naive_bayes(more_correlated,'in_college')
    col20.write(f'The accuracy of Naive Bayes: {naive_bayes_result3[1]}')
    col20.write(naive_bayes_result3[0])

    st.markdown('')
    st.markdown('The best results acheived from the Naive Bayes test was when all variables are included - it had an accuracy of 83.1%.')

if selected == 'Conclusion':
    st.title('Conclusion')
    st.markdown('In this case study, we exolored the different "college determinants" and how they correlate to the result of whether a student goes to college or not. We used a variety of visualizations to represent and compare variables, and used different algorithms such as chi squared and naive bayes to help us make predictions on the data. The final accuracy of our naive bayes prediction model was 83.1%, which is a relatively good result given our dataset had an equal distribution of the target variable.')
    



