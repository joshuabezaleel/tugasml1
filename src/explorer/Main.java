/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package explorer;

import classifier.decisiontree.myC45;
import classifier.decisiontree.myID3;
import classifier.neuralnetwork.MyMLP;
import classifier.neuralnetwork.MyPTR;
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
        processor.readDataset("data/Zoo/zoo.data");
//        processor.setFilterForANN();
        processor.buildClassifier(new myC45());
//        processor.buildFilteredClassifier(new Resample(), new myC45());
//        processor.percentageSplit_Eval(66);
        processor.nFoldCross_Eval(10);
//        processor.trainingSet_Eval();
//        processor.saveModel();
//        processor.loadModel("data/model/FilteredClassifier.model");
//        processor.readDataset("data/data_classifying/weather.nominal.arff");
//        processor.classifyDataset();
    }
    
}
