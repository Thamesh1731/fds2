import re
import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from pyresparser import ResumeParser
import src.notebook.skills_extraction as skills_extraction

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)
stopw = set(stopwords.words('english'))

# Load dataset
jd_df = pd.read_csv(r'C:\Users\Admin\ML_Projects\Job_Recommendation_System\Job-Recommendation-System\src\data\jd_structured_data.csv')

# Load the extracted resume skills
file_path = os.path.join('C:', 'Users', 'Admin', 'ML_Projects', 'Job_Recommendation_System',
                          'Job-Recommendation-System', 'utilities', 'resumes', 'CV.pdf')
skills = []
skills.append(' '.join(word for word in skills_extraction.skills_extractor(file_path)))

def ngrams(string, n=3):
    string = fix_text(string)  # fix text
    string = string.encode("ascii", errors="ignore").decode()  # remove non-ascii chars
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()  # normalize case - capital at start of each word
    string = re.sub(' +', ' ', string).strip()  # get rid of multiple spaces
    string = ' ' + string + ' '  # pad names for ngrams
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(skills)

nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
jd_test = (jd_df['Processed_JD'].values.astype('U'))

def getNearestN(query):
    queryTFIDF_ = vectorizer.transform(query)
    distances, indices = nbrs.kneighbors(queryTFIDF_)
    return distances, indices

# Get job recommendations based on the extracted skills
distances, indices = getNearestN(jd_test)
matches = []

for i, j in enumerate(indices):
    dist = round(distances[i][0], 2)
    matches.append([dist])

matches = pd.DataFrame(matches, columns=['Match confidence'])

# Recommend Top 5 Jobs based on candidate resume
jd_df['match'] = matches['Match confidence']
top_jobs = jd_df.head(5).sort_values('match')

# You can return or print top_jobs as needed
