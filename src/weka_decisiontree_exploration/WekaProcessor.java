/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka_decisiontree_exploration;

import java.io.File;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Rakhmatullah Yoga S, Joshua Bezaleel Abednego, Linda Sekawati
 */
public class WekaProcessor {
    //ATTRIBUTES
    private Instances learning_data;
    private Classifier classifier;
    private Random rand;
    
    //METHODS
    //Preprocess
    public Instances readFromCSV(String path) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        return loader.getDataSet();
    }
    
    public Instances readDataset(String path) throws Exception {
        Instances data;
        if(path.substring(path.length()-4).equalsIgnoreCase(".csv"))
            data = readFromCSV(path);
        else
            data = DataSource.read(path);
        if(data.classIndex()==-1)
            data.setClassIndex(data.numAttributes()-1);
        return data;
    }
    
    public void inputLearningData(String path) throws Exception {
        learning_data = readDataset(path);
    }
    
    //Classify
    public void buildClassifier(Classifier cls) throws Exception {
        classifier = cls;
        classifier.buildClassifier(learning_data);
    }
    
    public void fullTrainSet_Eval() throws Exception {
        Evaluation eval = new Evaluation(learning_data);
        eval.evaluateModel(classifier, learning_data); //use training set
        System.out.println(eval.toSummaryString("Evaluation results (Full Training)\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    
    public void nFoldCross_Eval(int folds) throws Exception {
        Evaluation eval = new Evaluation(learning_data);
        rand = new Random(1); // cross validation
        eval.crossValidateModel(classifier, learning_data, folds, rand); //cross validation
        System.out.println(eval.toSummaryString("Evaluation results ("+folds+" fold cross validation)\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    
    public void saveModel() throws Exception {
        SerializationHelper.write("data/model/"+classifier.getClass().getSimpleName()+".model", classifier);
    }
    
    public void loadModel(String modelPath) throws Exception {
        classifier = (Classifier) SerializationHelper.read(modelPath);
    }
        
    public void classifyDataset(String unlabelPath) throws Exception {
        String path_no_extension = unlabelPath.substring(0, unlabelPath.length()-5);
        Instances unclassified_dataset = readDataset(unlabelPath);
        Instances classified_dataset = new Instances(unclassified_dataset);
        for(int i=0;i< unclassified_dataset.numInstances(); i++) {
            double clsLabel = classifier.classifyInstance(unclassified_dataset.instance(i));
            classified_dataset.instance(i).setClassValue(clsLabel);
        }
        ConverterUtils.DataSink.write(path_no_extension+"-labeled.arff", classified_dataset);
    }
}
