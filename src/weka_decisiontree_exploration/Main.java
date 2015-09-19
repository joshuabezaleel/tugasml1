/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration;

import weka.classifiers.bayes.NaiveBayes;

/**
 *
 * @author Rakhmatullah Yoga S, Joshua Bezaleel Abednego, Linda Sekawati
 */
public class Main {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        WekaProcessor processor = new WekaProcessor();
        processor.inputLearningData("data/data_train/weather.nominal.arff");
        processor.buildClassifier(new NaiveBayes());
        processor.fullTrainSet_Eval();
        processor.nFoldCross_Eval(10);
        processor.saveModel();
        //processor.loadModel("");
        processor.classifyDataset("data/data_classifying/weather.nominal.arff");
    }
    
}
