/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package explorer;

import java.io.File;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.SimpleFilter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Rakhmatullah Yoga S, Joshua Bezaleel Abednego, Linda Sekawati
 */
public class WekaProcessor {
    //ATTRIBUTES
    private Instances dataset;
    private Classifier classifier;
    private Random rand;
    private String data_filepath;
    
    //METHODS
    //Preprocess
    public Instances readFromCSV(String path) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        return loader.getDataSet();
    }
    
    public void readDataset(String path) throws Exception {
        if(path.substring(path.length()-4).equalsIgnoreCase(".csv")) {
            data_filepath = path.substring(0, path.length()-4);
            dataset = readFromCSV(path);
        }
        else {
            data_filepath = path.substring(0, path.length()-5);
            dataset = DataSource.read(path);
        }
        if(dataset.classIndex()==-1)
            dataset.setClassIndex(dataset.numAttributes()-1);
    }
    
    public void removeAttribute(String idxAttr) throws Exception {
        Remove remove = new Remove();
        Instances data = new Instances(dataset);
        remove.setAttributeIndices(idxAttr);
        remove.setInputFormat(data);
        dataset = SimpleFilter.useFilter(data, remove);
    }
    
    //Classify
    public void buildFilteredClassifier(Filter filter, Classifier cls) throws Exception {
        FilteredClassifier filtercls = new FilteredClassifier();
        filtercls.setClassifier(cls);
        filtercls.setFilter(filter);
        classifier = filtercls;
        classifier.buildClassifier(dataset);
    }
    
    public void buildClassifier(Classifier cls) throws Exception {
        classifier = cls;
        classifier.buildClassifier(dataset);
    }
    
    public void trainingSet_Eval() throws Exception {
        Evaluation eval = new Evaluation(dataset);
        eval.evaluateModel(classifier, dataset);
        System.out.println(eval.toSummaryString("Evaluation results (given training set)\n", false));
        if(dataset.classAttribute().isNominal()) {
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
            System.out.println(eval.toMatrixString());
        }
    }
    
    public void percentageSplit_Eval(int percentage) throws Exception {
        int trainSize = (int) Math.round(dataset.numInstances()* percentage/100);
        int testSize = dataset.numInstances() - trainSize;
        dataset.randomize(new Random(1));
        Instances train = new Instances(dataset, 0, trainSize);
        Instances test = new Instances(dataset, trainSize, testSize);
        Classifier percent_cls = Classifier.makeCopy(classifier);
        Evaluation eval;
        percent_cls.buildClassifier(train);
        eval = new Evaluation(train);
        eval.evaluateModel(percent_cls, test);
        System.out.println(eval.toSummaryString("Evaluation results (Percentage split)\n", false));
        if(dataset.classAttribute().isNominal()) {
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
            System.out.println(eval.toMatrixString());
        }
    }
    
    public void nFoldCross_Eval(int folds) throws Exception {
        Evaluation eval = new Evaluation(dataset);
        rand = new Random(1); // cross validation
        eval.crossValidateModel(classifier, dataset, folds, rand); //cross validation
        System.out.println(eval.toSummaryString("Evaluation results ("+folds+" fold cross validation)\n", false));
        if(dataset.classAttribute().isNominal()) {
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
            System.out.println(eval.toMatrixString());
        }
    }
    
    public void saveModel() throws Exception {
        File file = new File("data/model/");
        if(!file.exists())
            file.mkdirs();
        SerializationHelper.write("data/model/"+classifier.getClass().getSimpleName()+".model", classifier);
    }
    
    public void loadModel(String modelPath) throws Exception {
        classifier = (Classifier) SerializationHelper.read(modelPath);
    }
    
    public void classifyDataset() throws Exception {
        Instances classified_dataset = new Instances(dataset);
        for(int i=0;i< dataset.numInstances(); i++) {
            double clsLabel = classifier.classifyInstance(dataset.instance(i));
            classified_dataset.instance(i).setClassValue(clsLabel);
        }
        ConverterUtils.DataSink.write(data_filepath+"-labeled.arff", classified_dataset);
    }
}
