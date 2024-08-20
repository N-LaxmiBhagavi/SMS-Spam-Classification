# SMS-Spam-Classification
This project focuses on classifying SMS messages as either spam or ham (non-spam) using machine learning techniques. The notebook guides you through data preprocessing, feature extraction, model training, and evaluation, offering a comprehensive approach to text classification.

### Project Structure
- __Data Loading and Exploration:__ Load the SMS dataset and perform initial exploratory analysis to understand its structure.
- __Data Visualization:__ Utilize visual representations to gain insights into the distribution of spam and ham messages.
- __Data Preprocessing:__ Clean and transform the data, and create additional features to enhance model performance.
- __Feature Extraction:__ Convert SMS text into numerical features using techniques like TfidfVectorizer.
- __Model Training:__ Train machine learning models such as Multinomial Naive Bayes and Decision Tree to classify SMS messages.
- __Model Evaluation:__ Evaluate model performance using various metrics and compare their effectiveness.
- __Predictions:__ Use the trained models to predict whether new SMS messages are spam or ham.                                                                                                                           
### Requirements
__Ensure you have the following Python packages installed:__ <br>

     pip install numpy pandas matplotlib seaborn nltk scikit-learn

---
The required Python packages are:
- __numpy:__ For numerical operations.
- __pandas:__ For data manipulation and analysis.
- __matplotlib:__ For plotting and data visualization.
- __seaborn:__ For advanced data visualizations.
- __nltk:__ For natural language processing tasks, such as stopword removal and text lemmatization.
- __scikit-learn (sklearn):__
                              - __TfidfVectorizer:__ For converting text into numerical features.
                              - __cross_val_score, train_test_split:__ For model evaluation and splitting data.
                              - __MultinomialNB:__ For building the Naive Bayes classification model.
                              - __DecisionTreeClassifier__: For building the decision tree model.
                              - __classification_report, confusion_matrix, accuracy_score:__ For evaluating model performance.
- __re:__ For regular expression operations to clean the text data.
- __string:__ For string manipulation.
- __warnings:__ For filtering out warnings during execution.
- Additionally, ensure the NLTK data for stopwords and WordNet lemmatizer is downloaded: <br>


           - import nltk
           - nltk.download("stopwords")
           - nltk.download("wordnet")
### Steps to Perform SMS Spam Classification:
- __Data Loading:__ Load the SMS dataset and explore its structure and content.
- __Data Visualization:__ Create count plots, histograms, and other visualizations to understand the distribution of spam and ham messages.
- __Data Preprocessing:__ Clean SMS text, handle class imbalance, and create new features such as word count, presence of currency symbols, and numeric digits.
- __Feature Extraction:__ Transform the SMS text into numerical features using TfidfVectorizer or similar methods.
- __Model Training:__ Train classification models, such as Multinomial Naive Bayes and Decision Tree, on the processed data.
- __Model Evaluation:__ Evaluate the models using accuracy, confusion matrix, classification report, and other relevant metrics.
- __Make Predictions:__ Apply the trained models to classify new SMS messages as spam or ham.
### Results
- The models effectively classify SMS messages with high accuracy.
- Data visualization provides key insights into features that distinguish spam from ham.
### Limitations
- Model performance is influenced by the quality and diversity of the dataset.
- Addressing class imbalance remains challenging and may require more advanced techniques.
### Conclusion
This project demonstrates a practical approach to SMS spam classification, showcasing the power of machine learning in text classification tasks. The methods and models used here provide a solid foundation for further development and real-world applications.

### License
This project is licensed under the MIT License.
