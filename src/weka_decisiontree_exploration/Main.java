/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration;

import weka.filters.supervised.instance.Resample;

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
        processor.readDataset("data/data_train/weather.nominal.arff");
        //processor.buildClassifier(new myC45());
        processor.buildFilteredClassifier(new Resample(), new myC45());
        processor.percentageSplit_Eval(66);
        //processor.nFoldCross_Eval(10);
        processor.saveModel();
        //processor.loadModel("");
        //processor.readDataset("data/data_classifying/weather.nominal.arff");
        //processor.classifyDataset();
    }
    
}
