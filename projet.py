from math import sqrt
import utils
from functools import reduce	# used in nbParams
import operator					# used in nbParams
import pandas as pd
import scipy
import matplotlib.pyplot as plt

# ##################### QUESTION 01 ##################### #

def getPrior(x):
	estimation = x['target'].mean()
	sigma=sqrt(estimation*(1-estimation))
	alpha = 1.96/sqrt(len(x)) * sigma
	return {
		"estimation":estimation,
		"min5pourcent":estimation - alpha,
		"max5pourcent":estimation + alpha
	}

# ##################### QUESTION 02 ##################### #

class APrioriClassifier(utils.AbstractClassifier):
	def __init__(self, df):
		pass
		
	def estimClass(self, attrs):
		return 1

	def statsOnDF(self, df):
		retourne = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0 , 'Precision': 0 , 'Rappel': 0 }
		ligne = 0
		for t in df.itertuples():
			dic = t._asdict()
			v = dic['target']
			attrs = utils.getNthDict(df, ligne)
			classePrevue = self.estimClass(attrs)

			if v == 1 and classePrevue == 1:
				retourne['VP'] += 1
			elif v == 0 and classePrevue == 0:
				retourne['VN'] += 1
			elif v == 0 and classePrevue == 1:
				retourne['FP'] += 1
			else:
				retourne['FN'] += 1
			ligne += 1

		retourne['Precision'] = retourne['VP'] / (retourne['VP'] + retourne['FP'])
		retourne['Rappel'] = retourne['VP'] / (retourne['VP'] + retourne['FN'])
		return retourne

# ##################### QUESTION 03 ##################### #

def P2D_l(df,attr):
	retourne = dict()
	
	for target in df['target'].unique():
		tmp = dict()
		for val_attr in df[attr].unique():
			tmp[val_attr] = 0
		retourne[target] = tmp

	size_of_df = df.groupby('target')[attr].count()

	for t in df.itertuples():
		dictio = t._asdict()
		target = dictio['target']
		attribut = dictio[attr]
		retourne[target][attribut] += 1
	
	for target in retourne.keys():
		for val_attribut in retourne[target].keys():
			retourne[target][val_attribut] /= size_of_df[target]

	return retourne

def P2D_p(df,attr):
	retourne = dict()
	
	for val_attr in df[attr].unique():
		tmp = dict()
		for target in df['target'].unique():
			tmp[target] = 0
		retourne[val_attr] = tmp
	
	size_of_df = df.groupby(attr)['target'].count()

	for t in df.itertuples():
		dictio = t._asdict()
		target = dictio['target']
		attribut = dictio[attr]
		retourne[attribut][target] += 1
	
	for val_attribut in df[attr].unique():
		for target in df['target'].unique():
			retourne[val_attribut][target] /= size_of_df[val_attribut]

	return retourne
	
class ML2DClassifier(APrioriClassifier):

	def __init__(self, df, attr):
		self.df = df
		self.attr = attr
		self.P2D_l = P2D_l(df,self.attr)

	def estimClass(self, attrs):
		val_attr = attrs[self.attr]
		list_proba = []
		list_key = list(self.P2D_l.keys()) # [1,0]
		list_key.reverse()
		for i in list_key:
			list_proba.append(self.P2D_l[i][val_attr])

		target = list_proba.index(max(list_proba))
		return target
			

class MAP2DClassifier(APrioriClassifier):

	def __init__(self, df, attr):
		self.df = df
		self.attr = attr
		self.P2D_p = P2D_p(df,self.attr)

	def estimClass(self, attrs):
		val_attr = attrs[self.attr]
		list_proba = []
		list_key = list(self.P2D_p[val_attr].keys()) # [0,1,2,3] for thal
		list_key.reverse()
		for i in list_key:
			list_proba.append(self.P2D_p[val_attr][i])

		target = list_proba.index(max(list_proba))
		return target

# ##################### QUESTION 04 ##################### #

def nbParams(df,list_attr=None):
	TAILLE_1_VAL = 8
	taille_totale = 0
	list_nb_val_attribut = []

	if list_attr == None:
		list_attr = list(df)

	for attribut in list_attr:
		list_nb_val_attribut.append(len(df[attribut].unique()))

	nb_valeurs_attribut_total = reduce(operator.mul, list_nb_val_attribut, 1)
	taille_totale = nb_valeurs_attribut_total * TAILLE_1_VAL

	nb_variables = len(list_attr)

	print ("{} variable(s) : {} octets".format(nb_variables , taille_totale))
	
def nbParamsIndep(df):
	TAILLE_1_VAL = 8
	taille_totale = 0
	list_attr = list(df)

	for attribut in list_attr:
		nb_valeurs_attribut = len(df[attribut].unique())
		taille_totale += nb_valeurs_attribut * TAILLE_1_VAL

	nb_variables = len(list_attr)

	print ("{} variable(s) : {} octets".format(nb_variables , taille_totale))

# ##################### QUESTION 05 ##################### #

def drawNaiveBayes(df,nom_attribut_classe):
	chaine_draw = nom_attribut_classe
	list_attr = list(df)
	list_attr.remove(nom_attribut_classe)
	for attribut in list_attr:
		chaine_draw += "->" + attribut + ";"
		chaine_draw += nom_attribut_classe
	return utils.drawGraph(chaine_draw)
	
def nbParamsNaiveBayes(df,nom_attribut_classe,list_attr = None):
	TAILLE_1_VAL = 8
	taille_totale = 0

	if list_attr == []: 
		nb_variables = 0
		taille_totale = len(df[nom_attribut_classe].unique()) * TAILLE_1_VAL

	else:
		if list_attr == None:
			list_attr = list(df)

		nb_variables = len(list_attr)
		list_nb_val_attribut = []

		for attribut in list_attr:
			list_nb_val_attribut.append(len(df[attribut].unique()))

		list_taille_table = [nb_val_attribut * len( df[nom_attribut_classe].unique() ) * TAILLE_1_VAL for nb_val_attribut in list_nb_val_attribut]

		taille_totale = sum(list_taille_table) - ( len(df[nom_attribut_classe].unique()) * TAILLE_1_VAL ) # because P(target|target) = P(target)

	print ("{} variable(s) : {} octets".format(nb_variables , taille_totale))
	
	
class MLNaiveBayesClassifier(APrioriClassifier):
	def __init__(self, df):
		self.df = df
		self.proba = dict()
		liste = list(df.columns.values)
		liste.remove("target")
		for attr in liste:
			self.proba[attr] = P2D_l(df, attr)
	
	def estimClass(self, attrs):
		res = self.estimProbas(attrs)
		if res[0] >= res[1]:
			return 0
		return 1
		
	def estimProbas(self, attrs):
		out1 = 1
		out2 = 1
		for p in self.proba:
			temp = self.proba[p]
			if attrs[p] in temp[0]:
				out1 *= temp[0][attrs[p]]
				out2 *= temp[1][attrs[p]]
			else:
				return {0: 0.0, 1: 0.0}
		return {0: out1, 1: out2}


class MAPNaiveBayesClassifier(APrioriClassifier):
	def __init__(self, df):
		self.df=df
		self.moy = self.df["target"].mean()
		self.proba = dict()
		liste = list(df.columns.values)
		liste.remove("target")
		for attr in liste:
			self.proba[attr] = P2D_l(df, attr)
	
	def estimClass(self, attrs):
		res = self.estimProbas(attrs)
		if res[0] >= res[1]:
			return 0
		return 1
		

	def estimProbas(self, attrs):
		out2 = self.moy
		out1 = 1-self.moy
		
		for p in self.proba:
			temp = self.proba[p]
			if attrs[p] in temp[0]:
				out1 *= temp[0][attrs[p]]
				out2 *= temp[1][attrs[p]]
			else:
				return {0: 0.0, 1: 0.0}

		return {0: (out1 / (out1 + out2)), 1: (out2 / (out1 + out2))}	

# ##################### QUESTION 06 ##################### #

def isIndepFromTarget(df,attr,x):
	contingence = pd.crosstab(df[attr],df.target).values
	chi2, p, dof, expected = scipy.stats.chi2_contingency(contingence)
	return not(p<x)
	
class ReducedMLNaiveBayesClassifier(APrioriClassifier):
	def __init__(self, df, seuil):
		self.df = df
		self.proba = dict()
		liste = list(df.columns.values)
		liste.remove("target")
		for attr in liste:
			if not isIndepFromTarget(df, attr, seuil):
				self.proba[attr] = P2D_l(df, attr)
			

	def estimClass(self, attrs):
		res = self.estimProbas(attrs)
		if res[0] >= res[1]:
			return 0
		return 1


	def estimProbas(self,attrs):
		out1 = 1
		out2 = 1
		for p in self.proba:
			temp = self.proba[p]
			if attrs[p] in temp[0]:
				out1 *= temp[0][attrs[p]]
				out2 *= temp[1][attrs[p]]
			else:
				return {0: 0.0, 1: 0.0}
		return {0: out1, 1: out2}

	def draw(self):
		out = ""
		for i in self.proba.keys():
			out += "target"+"->"+i+";"
		return utils.drawGraph(out)



class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
	def __init__(self, df, seuil):
		self.df=df
		self.moy = self.df["target"].mean()
		self.proba = dict()
		liste = list(df.columns.values)
		liste.remove("target")
		for attr in liste:
			if not isIndepFromTarget(df, attr, seuil):
				self.proba[attr] = P2D_l(df, attr)



	def estimClass(self, attrs):
		res = self.estimProbas(attrs)
		if res[0] >= res[1]:
			return 0
		return 1


	def estimProbas(self, attrs):
		out2 = self.moy
		out1 = 1-self.moy
		
		for p in self.proba:
			temp = self.proba[p]
			if attrs[p] in temp[0]:
				out1 *= temp[0][attrs[p]]
				out2 *= temp[1][attrs[p]]
			else:
				return {0: 0.0, 1: 0.0}

		return {0: (out1 / (out1 + out2)), 1: (out2 / (out1 + out2))} 

	def draw(self):
		out = ""
		for i in self.proba.keys():
			out += "target"+"->"+i+";"
		return utils.drawGraph(out)
		
# ##################### QUESTION 07 ##################### #

def mapClassifiers(dic, df):
	precision = []
	rappel = []
	name = []
	
	for i, n in dic.items():
		 dico_stats = n.statsOnDF(df)
		 name.append(i)
		 precision.append(dico_stats["Precision"])
		 rappel.append(dico_stats["Rappel"])
	
	plt.scatter(precision, rappel, color="r", marker="x")
	for i, txt in enumerate(name):
		plt.annotate(txt, (precision[i], rappel[i]))
	
	plt.show()
