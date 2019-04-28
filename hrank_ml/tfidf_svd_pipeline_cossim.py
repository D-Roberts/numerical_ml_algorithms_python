
import os
import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD


def load_data(filename):
	questions = []
	with open(filename, 'r') as f:
		sentences = f.readline().strip().split('.')
		for i in range(5):
			line = f.readline().strip()
			questions.append(line)
		line = f.readline()
		answers = line.strip().split(';')
	return np.array(sentences), np.array(questions), np.array(answers)

def print_answers(sentences, questions, answers):
	p = Pipeline([('tfid', TfidfVectorizer(lowercase=True,
						  strip_accents='ascii',
						  stop_words='english')),
		('lsa', TruncatedSVD(n_components=20))]) # overkill for such small - but improved score to 5/5

	stf = p.fit_transform(sentences) 
	ans = p.transform(answers)
	# print(ans.toarray().shape)
	qs = p.transform(questions)
	print(qs.shape)

	# get pairwise similarities
	cs_sa = cs(stf, ans)
	print(cs_sa.shape)
	cs_sq = cs(stf, qs)
	print(cs_sq.shape)

	# get top correl sent for each q
	top_sq = np.argmax(cs_sq, axis=0)
	print(top_sq)
	for j in range(len(top_sq)):
		# print(sentences[i])
		print('question is: ', questions[j])
		# find the top correl answers to the sent
		print('answer is: ', answers[np.argmax(cs_sa[top_sq[j]])])
		print('\n')



def main():
	path = '/Users/denisaroberts/Desktop/matching-questions-answers-testcases/input'
	files = os.listdir(path)
	# print(files)
	sentences, questions, answers = load_data(os.path.join(path, 
														  'input00.txt'))
	print_answers(sentences,
				  questions,
				  answers)


if __name__ == '__main__':
	main()