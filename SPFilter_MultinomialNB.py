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
        hamCount = 0
        spamCount = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                hamCount += 1
            else:
                spamCount += 1

        class_log_prior[HAM] = math.log(hamCount)
        class_log_prior[SPAM] = math.log(spamCount)

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
        ham = np.zeros(most_common_word)
        spam = np.zeros(most_common_word)

        for row in range(len(features)):
            for col in range(most_common_word):
                if row <= hamCount:
                    ham[col] += features[row][col]
                    hamSum += features[row][col]
                else:
                    spam[col] += features[row][col]
                    spamSum += features[row][col]
        for i in range(most_common_word):
            spam[i] += smooth_alpha
            ham[i] += smooth_alpha
        hamSum += most_common_word*smooth_alpha
        spamSum += most_common_word*smooth_alpha


        for j in range(most_common_word):
            feature_log_prob[HAM][j] = math.log(ham[j]/hamSum)
            feature_log_prob[SPAM][j] = math.log(spam[j]/spamSum)









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
