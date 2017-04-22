# Machine Learning Engineer Nanodegree
## Capstone Project
Yudai Furukawa  
March 31st, 2017

## I. Definition
_(approx. 1-2 pages)_

### Project Overview
This project is about stock investing, and I am especially focusing on price prediction of a stock market index. A stock index is an aggregate value produced by combining several stocks, and it helps investors to measure and compare values of the stock markets such as in the US and Japan.
The Dow Jones Industrial Average (DJIA), Nasdaq Composite index and the S&P Composite are examples of stock indices.
As a wealth of information such as price, earnings, dividends, and CPI are available, we are going to use those information to do the prediction.  
A dataset of S&P Composite published by Yale Department of Economics will be used in this project. For more infomation, please refer the link below:
https://www.quandl.com/data/YALE-Yale-Department-of-Economics

In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

### Problem Statement
For this project, the task is to build a stock index price predictor. A 12 month forward price change of S&P composite will be predicted by using regression. The project is going to be a supervised learning.

Following steps will be taken to make the predictor.
1. Figuring out what inputs are necessary to predict a 12 month forward price change by using correlation between each input and 12 month forward price changes
    * For example, earning growth and PER will be major inputs as those are considered leading indicators to predict a stock price.
2. Figuring out the best regressor by trying a multiple regressors. 
    * Coefficient of determination will be used to measure performances of regressors. 


In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

### Metrics
The coefficient of determination (the r2 score) will be used for scoring the result of the prediction in this task. 
The r2 score provides a measure of how well the regression line represents the data. However, it has a weakness as the score could be greatly affected by unusual data points.
Since the problem of this project is regression, the r2 score will be sufficient for this project as long as outliers are omitted.


In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
A dataset of S&P Composite published by Yale Department of Economics will be used in this project. 
The dataset (named snp in this project) is monthly time series of S&P Composite Price, Dividend, Earnigns, CPI, Long Interest Rate, Real S&P Composite Price, Real Dicidend, Real Earnings, and Cyclically Adjusted PE Ration since 1831-1-31 up to date.

For more infomation for calculation methodology of each factor, please refer the link below:
https://www.quandl.com/data/YALE-Yale-Department-of-Economics

As of 2017-04-08, the basic statistics of snp the dataset is following.

| Statistics | S&P Composite | Dividend  | Earnings          | CPI |       Long Interest Rate  | Real Price | Real Dividend | Real Earnings |        Cyclically Adjusted PE Ratio  |    
| -------------------- | :---------------: |  :---------------: | :---------------: | :---------------: | :---------------: | :---------------: | :---------------: |  :---------------: |  ---------------: | 
| count    | 1756.000000  | 1755.000000  | 1749.000000  | 1756.000000 |  1756.000000 | 1756.000000  |  1755.000000   | 1749.000000   |1636.000000  |
|mean  |    242.537415   |  5.344903  |  12.046968  |  56.433670 |   4.584025  | 482.828989  |    14.590266  |    28.558423  | 16.748036  |
|std |     478.579184 |    9.010165   | 22.474642   | 69.298402 |  2.290630   |505.897801    |   7.846696    |  22.451132  |  6.649515  |
|min   |      2.730000    | 0.180000  |   0.160000  |   6.279613 |   1.500000  |  66.095494    |   4.870418    |   4.093124    | 4.784241 |
| 25%   |      7.680000   |       NaN  |        NaN   | 10.100000 |          NaN   |  3.308333 |  165.900151    |        NaN     |      NaN   |
|50%        |16.005000  |        NaN   |       NaN  |  18.100000  |     3.870000 |  246.207502    |        NaN     |       NaN     |   NaN  |
| 75%     |  115.550000|          NaN    |     NaN|    84.200000 |     5.240000 |  588.255364 |           NaN     |       NaN |            NaN  |
|max |     2357.000000 |   46.380000  | 105.960000  | 244.176000 |      15.320000 | 2357.000000  |    46.416308  |   108.695460    |  44.197940 |

As you can see some of the cells are filled by NaN as some data are missing in the dataset. Also, because when dealing with economical data, inflation has to be carefully taken into account as CPI tends to grow overtime and values of price and earnings tend to have smaller values in the past. Therefore, only real values, Long Interest Rate, and Cyclicall Adjusted PE Ratio in the previous table can be taken seriously in statistical analysis without any modification.





In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
snp_figure1 (timeSeries)
snp_figure2 (timeSeries)

Figure 1 is a time series plot of the raw dataset. Figure 2 is also time series plot without "Real Price" and "S&P Composite" in order to visualize other features clearly. Time series plot was chosen as the data itself is a time series. For the both figures, the horizontal axis displays years and the vertical axis displays figures for each feature where the unit is different for each feature.

In figure 1, you can clearly see both "Real Price" and "S&P Composite" have been increasing in general as time goes. Therefore, it would be better nomarizing data when doing analysis. Also, in figure 2, you can see the same trends besides for "Cyclically adjusted PE Ration" and "Long Interest Rate". Therefore, for the same reasons, it would be better nomarizing data when doing analysis besides those two features.


In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
As discussed, the original dataset needs to be nomalized in order to predict % change of S&P Composite price with a return horizon of 1 year (named snp_changes). I first removed real values as this project aims to predict change of nominal S&P Composite price and CPI in the dataset allows us to calculate the real value when needed. After this, instead of using raw dataset, I'm going to use 12 months changes for each feature plus the original figures of "Cyclically adjusted PE Ration" and "Long Interest Rate". The target feature will be 12 month forward changes in S&P Composite price.
After nomalizing, some features.

In this project, following regressors will be used in predicting the 12 month forward price change in S&P Composite.
1. Linear Regressor
2. KNeibors
3. SVR

I chose the linear regressor as 
I chose KNeibors as 
I chose SVR as 

In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms 	and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
As discussed, the dataset needs to be nomalized in order to predict % change of S&P Composite price with a return horizon of 1 year (named snp_changes). Therefore, I conducted following modification on the dataset. 

1. Removed factors with real values as the project is focusing on predicting % change of nominal S&P Composite price.
2. Created a dataset called snp_changes which shows 1 year change of each factor in order to see the relationship of changes of 1 yeare S&P Composite price and other factors.
3. Added a target factor "y" which is 12 months forward return of S&P Composite.
4. Added following factors that seem to be have influence on the prediction.
    * The real value of PE Ratio as the ratio is considered as a good indicator predict return. 

As of 2017-04-08, the basic statistics of the snp_changes dataset is following.

| statistics|    S&P Composite (%change)|   Dividend  (%change)|   Earnings (%change) | Cyclically Adjusted PE Ratio (%change) | y (%change) | PE Ratio (value) |
| -------------------- | :---------------: |  :---------------: | :---------------: | :---------------: | :---------------:| ---------------: | 
|count  |  1629.000000 | 1629.000000 | 1629.000000  | 1629.000000 | 1629.000000   |    1629.000000  |    
|mean     |   0.062039   |  0.044323  |   0.093620               |       0.021502  | 0.061958     |    16.734102      |
|std     |    0.187358    |0.106872  |   0.537744   |                   0.192371 |0.187298    |      6.655788   |
|min     |   -0.656092 |   -0.390244    |-0.886405|                     -0.630535 | -0.656092     |     4.784241   |
|25%   |     -0.060241   |  0.000000  |  -0.076412               |      -0.098145 | -0.060241    |     11.754449|  
|50%     |    0.068465    | 0.047917 |    0.053571 |                     0.020997|   0.068465 |        16.061148  |
|75%    |     0.186898 |    0.097610  |   0.173666 |                     0.137546   | 0.185828    |     20.384150|
|max     |    1.241517   |  0.531915  |   7.934754 |                     1.355688 |  1.241517     |    44.197940 |




In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
