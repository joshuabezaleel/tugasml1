/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import weka.classifiers.Classifier;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.Range;

/**
 *
 * @author asus
 */
public class MyCrossValidateModel {
    public int[][] crossValidateModel(Classifier classifier, Instances data,
    int numFolds, Random random)
    throws Exception {

    // Make a copy of the data we can reorder
    data = new Instances(data);
    data.randomize(random);
    if (data.classAttribute().isNominal()) {
      data.stratify(numFolds);
    }

    // We assume that the first element is a StringBuffer, the second a Range
    // (attributes
    // to output) and the third a Boolean (whether or not to output a
    // distribution instead
    // of just a classification)
//    if (forPredictionsPrinting.length > 0) {
//      // print the header first
//      StringBuffer buff = (StringBuffer) forPredictionsPrinting[0];
//      Range attsToOutput = (Range) forPredictionsPrinting[1];
//      boolean printDist = ((Boolean) forPredictionsPrinting[2]).booleanValue();
//      printClassificationsHeader(data, attsToOutput, printDist, buff);
//    }
    int[][] returnValue = new int[data.numClasses()][data.numClasses()];
    // Do the folds
    int [][] temp = null;
    for (int i = 0; i < numFolds; i++) {
      Instances train = data.trainCV(numFolds, i, random);
//      setPriors(train);
//      Classifier copiedClassifier = Classifier.makeCopy(classifier);
      classifier.buildClassifier(train);
      Instances test = data.testCV(numFolds, i);
      temp = evaluateModel(classifier, test);
      for(int j=0;j<data.numClasses();j++){
        for(int k=0;k<data.numClasses();k++){
            returnValue[j][k] = temp[j][k];
        }
    }
    
        
    }
//    m_NumFolds = numFolds;
    return returnValue;
  } 
    
    public int[][] evaluateModel(Classifier classifier, Instances test) throws Exception{
        int[][] confusionMatrix = new int[test.numClasses()][test.numClasses()];
        for(int i=0;i<test.numInstances();i++){
            int target = (int) test.instance(i).value(test.classIndex());
            int output = (int) Math.round(classifier.classifyInstance(test.instance(i)));
            confusionMatrix[target][output]++;
        }        
        return confusionMatrix;
    }
}
