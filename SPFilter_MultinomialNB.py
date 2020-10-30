import os
import math
import numpy as np


most_common_word = 3000
#avoid 0 terms in features
smooth_alpha = 1.0
class_num =2 #we have only two classes: ham and spam
class_log_prior = [0.0, 0.0]#probability for two classes
feature_log_prob = np.zeros((class_num, most_common_word))#feature parameterized probability
SPAM = 1 #spam class label
HAM = 0 #ham class label


class MultinomialNB_class:
    
    #multinomial naive bayes
    def MultinomialNB(self, features, labels):
        '''calculate class_log_prior

		* loop over labels
		* if the value of the term in labels = 1 then ham++
		* if the value of the term in labels = 0 then spam++
		* class_log_prior[0] = Math.log(ham)
		* class_log_prior[1] = Math.log(spam)
		'''
        ham = 0
        spam = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                ham += 1
            else:
                spam += 1

        class_log_prior[0] = math.log(ham)
        class_log_prior[1] = math.log(spam)

        '''
		//calculate feature_log_prob
		
		/**
		 * nested loop over features
		 * for row = features.length
		 *     for col = most_common
		 *         ham[col] + features[row][col]
		 *         spam[col] + features[row][col]
		 *         sum of ham
		 *         sum of spam
		 * for i = most_common
		 *     ham[i] + smooth_alpha
		 *     spam[i] + smooth_alpha
		 * sum of ham += most_common*smooth_alpha
		 * sum of spam += most_common*smooth_alpha
		 * for j = most_common
		 *     feature_log_prob[0] = Math.log(ham[i]/sum of ham)
		 *     feature_log_prob[1] = Math.log(spam[i]/sum of spam)
		 */'''
        hamSum = 0;
        spamSum = 0;
        for row in range(len(features)):
            for col in range(most_common_word):
                if row <= ham:
                    feature_log_prob[HAM][col] += features[row][col]
                    hamSum += feature_log_prob[HAM][col]
                else:
                    feature_log_prob[SPAM][col] += features[row][col]
                    spamSum += feature_log_prob[SPAM][col]
        for i in range(most_common_word):
            feature_log_prob[0][i] += smooth_alpha
            feature_log_prob[1][i] += smooth_alpha
        hamSum += most_common_word*smooth_alpha
        spamSum += most_common_word*smooth_alpha


        for j in range(most_common_word):
            feature_log_prob[0][j] = math.log(feature_log_prob[0][j]/hamSum)
            feature_log_prob[1][j] = math.log(feature_log_prob[1][j]/spamSum)









    #multinomial naive bayes prediction
    def MultinomialNB_predict(self, features):
        classes = np.zeros(len(features))


        '''/**
		 * nested loop over features with i and j
		 * calculate ham_prob and spam_prob
		 * add ham_prob and spam_prob with class_log_prior
		 * if ham_prob > spam_prob
		 * HAM
		 * else SPAM
		 * return  classes
		 */'''
        for i in range(len(features)):
            ham_prob = 0.0
            spam_prob = 0.0
            for j in range(len(features[i])):
                ham_prob += feature_log_prob[HAM][j] * features[i][j]
                spam_prob += feature_log_prob[SPAM][j] * features[i][j]
                ham_prob += class_log_prior[HAM]
                spam_prob += class_log_prior[SPAM]
            if ham_prob > spam_prob:
                classes[i] = HAM
            else:
                classes[i] = SPAM

        return classes
