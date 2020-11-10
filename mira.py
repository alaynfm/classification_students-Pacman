 # mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        BestC = Cgrid[0]#probar con el primer C
        BestCResult = float("-inf")
        weightsForC = []
        
        for C in Cgrid:
            # teneis que probar de las distintas C que hay en Cgrid la que es la mas apropiada
            # teneis que ir por todo el train y aprender los w como en el perceptron,
            # salvo que en este caso al actualizar w de las clases correspondientes (correcta y la predicha)
            # no hareis w = w +/- f(x) (segun sea la buena o no)
            # sino w = w +/- (Tau * f(x)
            # Para ello teneis que calcular Tau con la formula de las transparencias
            # (w(clase predecida) - w(clase buena)*f(x)+1.0)/2.0(f(x)*f(x))
            # Cuidado!! f(x)*f(x), emplead self.ff(item) para calcularlo donde item es f(x)
            # la teneis definida justo debajo
            #entrenar con un valor C concreto de entre    Cgrid = [0.002, 0.004, 0.008]
            # CUIDADO!! para entrenar empleo el trainingData
            
            "*** YOUR CODE HERE ***"

            #despues de entrenar con una C
            #por cada C calculo el numero de ejemplos del validation set
            #que ha clasificado bien con ese valor de C

            guesses = self.classify(validationData)#numero de aciertos empleando el valor de C actual

            #si el numero de aciertos es mayor que el mayor obtenido hasta el momento con las C previas
            #actualizo mi w mejor, la C mejor y el numero de aciertos de la mejor C hasta el momento
            if len(list(set(guesses).intersection(validationLabels))) >= BestCResult:
                BestCResult = len(list(set(guesses).intersection(validationLabels)))
                BestC = C
                weightsForC= self.weights.copy()

        self.weights = weightsForC
        util.raiseNotDefined()

    def ff (item):
        elemtWise = []
        for feat in self.features:
            val = item[feat] * item[feat]
            elemtWise.append(val)
        normOfItem = math.sqrt(sum(elemtWise))
        return(normOfItem)
    
    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


