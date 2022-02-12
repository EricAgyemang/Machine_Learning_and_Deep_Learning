# RESEARCH TOPIC:
  # APPLICATION OF LOGISTIC REGRESSION AND NEURAL NETWORK MODELS TO BREAST CANCER PATIENTS IN USA (MAT 490, ISU)
  
  # ABSTRACT
Breast cancer is a disease in which the cells in the breast grows out of control. It is the second
most common cause of death from cancer in women after lung cancer and over 50 thousand
new cases are reported every year in the United States. The survival rates of breast cancer depend on many factors. 
Using a sample of 100,002 breast cancer patients from three states in
the United States covering the period 1992 – 2019, this study focuses on developing predictive
models based on patient’s demographic risk factors, and mammographic descriptors to predict
breast cancer patient’s last status (Alive or dead) and examine the influence of these factors
on the patient’s last status. Using machine learning and deep learning techniques, five models
were developed using logistic regression, feed forward neural networks, and Random Forest
Classification to predict the breast cancer patient’s last status for both sex and each sex separately. We compared the output
of all models and found that the Convolutional Neural Network
outperformed all models in both sex and each sex separately. However, in assessing the influence of the risk factors on patient’s
last status, the Random Forest is proposed to be the best fit
compared with all the other model. This work provides insight into increasing the effectiveness
of machine learning in contributing to improve clinical decision making.
Keywords: Model Evaluation, Prediction Accuracy, Breast Cancer, Abnormal Cells , Clinical Outcomes.

# RESEARCH QUESTIONS
This project is guided by the following research question: 
   1. Which factors affect the survival and death rate of all breast cancer patients in the USA?
   2. Which factors affects male and female breast cancer patients survival and death rate separately in the USA?
   3. Which model is the best fit for estimating breast cancer risk factors in contributing to improve clinical decision making?
 
 # SIGNIFICANCE OF THE STUDY
This study does not only contributes to extent literature, it also contribute to creating the awareness, bringing to knowledge and understanding of 
key factors that affect breast cancer patients, serves as a guide to both patients and physicians, and intends to open doors to more appropriate means
for prevention, accurate method of diagnosis, better prediction of breast cancer outcomes, and treating breast cancer in the USA. These will help ensure
an improved quality of life of breast cancer patients which has a commanding effect of mitigating the mortality rate of the breast cancer patient.

# SCOPE OF STUDY

This study does not only contributes to extent literature, it also contribute to creating the awareness, bringing to knowledge 
and understanding of the factors that affect breast cancer patients, serving as a guide to both patients and physicians which intends 
to open the doors to a more appropriate means for prevention, accurate method of diagnosis, better prediction of clinical
outcomes, and treating breast cancer in the USA. These will help ensure an improved quality of life of breast cancer patients which has a
commanding effect of mitigating the mortality rate of the breast cancer patient.

# METHODOLOGY  
To answer the research questions posed in the study, data on breast cancer cases in San Francisco, Connecticut, and New
Jersey from the years 1992 to 2016 was taken from CDC website. These states are part of the top 10 states in the USA which experience high 
rate of cancer contraction. A total of 100,002 observations were collected and the variables collected include mammographic descriptors and
demographic risk factors. The dependent variable chosen for this study is the Patients Last Status (Alive or dead). All patients for the study 
are first analyzed together. Afterwards, the data is split into gender (Male and Female data) and analyze separately to get the predictions for
each gender.The models' performance are determined and compared; The logistic regression, MLP-Network(Deep learning), CNN, Random Forest classification,
and RBF predictions are obtained by the use of confusion matrix. The accuracy of prediction, Sensitivity Analysis(SA), Specificity Analysis, Positive Predictive 
Value(PPV), Negative Predictive Value(NPV), and the Receiver Operating Characteristic (ROC) curve were ascertained.

# DEFINITION OF MODEL EVALUATION APPROACHES USED.
  1. **Sensitivity Analysis:** This tells us the probability that the model predicts positive as death given
     that the patient actually died. Thus Prob(+ as death / patient actually died).
  2. **Positive Predicted Value (PPV) Analysis:** This tells us that, given the model predict positive
     as death on patients, what is the probability that they actually died? Thus Prob(patient actually died / + as death ).
  3. **Specificity Analysis:** This tells us the probability that the model predicts negative as Alive given
     that the patients are alive. Thus Prob(- as Alive / patient actually Alive).
  4. **Negative Predicted Value (NPV) Analysis:** This tells us that given that the patients are alive,
     what is the probability that the model predicts negative as death? Thus Prob( - as death / patient actually Alive).
  5. **The Receiver Operating Characteristic (ROC) curve:** in classifying breast cancer patients as
     alive or dead with use of receiver operating characteristic (ROC) curves, the area under a ROC
     curve (AUC) indicates how well a prediction model discriminates between healthy patients and
     patients who died. The value of an AUC varies between 0.5 (ie, random guess) and 1.0 (perfect
     accuracy). The higher the value of the AUC, the better the model.
     
# RESULTS AND KEY FINDINGS.
   - The detailed results and key findings are discussed in the project report document attached as part of the repository files
     
     
     



