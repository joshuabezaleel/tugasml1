/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import explorer.WekaProcessor;
import java.io.File;
import java.io.IOException;
import static java.lang.Double.compare;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import org.jblas.DoubleMatrix;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author joshua
 */
public class SingleLayerPerceptron extends Classifier {
    private double learningRate;
    private double threshold;
    private int maxIteration;
    private boolean isRandomWeight;
    private double initialWeight;
    private NominalToBinary nominalToBinaryFilter;
    private boolean isNormalize;
    private Normalize normalizeFilter;
    private Instances data;
    private int algorithm;
    private DoubleMatrix weight;
    private DoubleMatrix deltaWeight;
    private Attribute classAttribute;
    
    public SingleLayerPerceptron(){
        this.learningRate = 0.1;
        this.threshold = 0.0;
        this.maxIteration = 10;
        this.isNormalize = true;
        this.isRandomWeight = true;
    }
    
    public SingleLayerPerceptron(double learningRate, double threshold, int maxIteration){
        this.learningRate = learningRate;
        this.threshold = threshold;
        this.maxIteration = maxIteration;
        this.isNormalize = true;
        this.isRandomWeight = true;
    }
    
    public SingleLayerPerceptron(double learningRate, double threshold, int maxIteration, double initialWeight){
        this.learningRate = learningRate;
        this.threshold = threshold;
        this.maxIteration = maxIteration;
        this.isNormalize = true;
        this.isRandomWeight = true;
        this.initialWeight = initialWeight;
    }
    
    public double getLearningRate(){
        return this.learningRate;
    }
    
    public double getThreshold(){
        return this.threshold;
    }
    
    public int getMaxIteration(){
        return this.maxIteration;
    }
    
    public int getAlgorithm(){
        return this.algorithm;
    }
    
    public double getInitialWeight(){
        return this.initialWeight;
    }
    
    public boolean getIsNormalize(){
        return this.isNormalize;
    }
    
    public void setIsNormalize(boolean isNormalize){
        this.isNormalize = isNormalize;
    }
    public void setInitialWeight(double initialWeight){
        this.isRandomWeight = false;
        this.initialWeight = initialWeight;
    }
    
    public void setLearningRate(double learningRate){
        this.learningRate = learningRate;
    }
    
    public void setThreshold(double threshold){
        this.threshold = threshold;
    }
    
    public void setMaxIteration(int maxIteration){
        this.maxIteration = maxIteration;
    }
    
    public void setAlgorithm(int algorithm){
        this.algorithm = algorithm;
    }
    
    private void randomWeight(DoubleMatrix weight){
        Random rand = new Random();
        double min = -0.5;
        double max = 0.5;
        double randomValue;
        for(int i=0;i<weight.length;i++){
            randomValue = min + (max-min) * rand.nextDouble();
        }   
    }
    
    private double sum(Instance instance){
        double temp = 0.0;
        temp = temp + (1 * weight.get(0));
        for(int i=1;i<weight.length;i++){
            temp = temp + instance.value(i-1) * weight.get(i);
        }
        return temp;
    }
    
    private double getTarget(Instance instance, boolean nominal){
        double target = 0.0;
        
        if(nominal && this.algorithm == 1){
            if(compare(instance.value(instance.classAttribute()),1.0)==0){
                target = 1.0;
            } else if (compare(instance.value(instance.classAttribute()),0.0)==0){
                target = -1.0;
            }
        }else{
            target = instance.value(instance.classAttribute());
        }
        return target;
    }
    
    public Instances nominalToNumeric(Instances data) throws Exception{
        this.nominalToBinaryFilter = new NominalToBinary();
        this.nominalToBinaryFilter.setInputFormat(data);
        data = Filter.useFilter(data, this.nominalToBinaryFilter);
        return data;
    }
    
    @Override
    public Capabilities getCapabilities() { 
        Capabilities result = super.getCapabilities();
        result.disableAll();
        
        //attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        
        //class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.DATE_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        
        return result;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.data = data;
        
        //Preprocessing
        getCapabilities().testWithFail(data);
        data = new Instances(data);
        data.deleteWithMissingClass();
            
        //Delete missing values
        Enumeration attributes = data.enumerateAttributes();
        while(attributes.hasMoreElements()){
            Attribute attribute = (Attribute) attributes.nextElement();
            data.deleteWithMissing(attribute);
        }

        data = nominalToNumeric(data);

        if (isNormalize) {
            normalizeFilter = new Normalize();
            normalizeFilter.setInputFormat(data);
            data = Filter.useFilter(data, normalizeFilter);
        }

        this.classAttribute = data.classAttribute();

        this.weight = new DoubleMatrix(1, data.numAttributes());
        this.deltaWeight = new DoubleMatrix(1, data.numAttributes());

        if(isRandomWeight){
            randomWeight(weight);
        }else{
            for(int i = 0; i < this.weight.length; i++){
                this.weight.put(i, this.initialWeight);
        }

        int epoch = 1;
        double mse = Double.POSITIVE_INFINITY;

        do{
            for(int it=0;it<data.numInstances();it++){
                Instance instance = data.instance(it);

                double sum = this.sum(instance);
                double output = 0.0;

                if(this.algorithm == 1){
                    output = signFunction(sum);	
                }else{
                    output = sum;
                }	

                double target = this.getTarget(instance, instance.classAttribute().isNominal());
                double error = target - output;

                for(int i=0;i<instance.numAttributes();i++){
                        if(i == 0){
                            if(this.algorithm != 2){
                                deltaWeight.put(i, this.learningRate * error * 1);
                            }else{
                                deltaWeight.put(i, deltaWeight.get(i) + error * 1);
                            }
                        }else{
                            if(this.algorithm != 2){
                                deltaWeight.put(i, this.learningRate * error * instance.value(i - 1));
                            }else{
                                deltaWeight.put(i, deltaWeight.get(i) + error * instance.value(i - 1));
                            }
                        }
                }
                if(this.algorithm != 2)
                        weight.addi(deltaWeight);
            }   

            if(this.algorithm == 2){
                deltaWeight.muli(this.learningRate);
                weight.addi(deltaWeight);
                deltaWeight = DoubleMatrix.zeros(deltaWeight.rows, deltaWeight.columns);
            }

            double squaredError = 0.0;
            for(int i=0;i<data.numInstances();i++){
                Instance instance = data.instance(i);
                double sum = this.sum(instance);
                double output = 0.0;

                if(this.algorithm == 1)
                        output = signFunction(sum);
                else
                        output = sum;

                double target = this.getTarget(instance, instance.classAttribute().isNominal());
                double error = target - output;

                squaredError += Math.pow(error, 2.0);
            }
            mse = squaredError / 2.0;
            epoch++;
        }while(epoch<this.maxIteration&&compare(mse,this.threshold)>=0);
    }
    }
    
    public double signFunction(double sum){
        double temp=0;
        if(sum>=0){
            temp = 1;
        }else{
            temp = -1;
        }
        return temp;
    }
    
    private double countOutput(double value, double low, double up){
        double temp;
        if(compare(value,low)<0){
            temp = low;
        }else if(compare(value,up)>0){
            temp = up;
        }else if(compare(Math.abs(low-value),Math.abs(up-value))>=0){
            temp = up;
        }else{
            temp = low;
        }
        return temp;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        double[] outputs = new double[instance.numClasses()];
        if(instance.classAttribute().isNumeric()){
            outputs[0] = this.classifyInstance(instance);
        }else if(instance.classAttribute().isNominal()){
            outputs[(int) classifyInstance(instance)] = 1.0;
        }
        
        return outputs;
    }
    
    public double classifyInstance(Instance instance) throws Exception{
        Instance result = instance;
        
        nominalToBinaryFilter.input(instance);
        result = nominalToBinaryFilter.output();
        
        if(isNormalize){
            normalizeFilter.input(result);
            result = normalizeFilter.output();
        }
        
        double sum = this.sum(result);
        double output = 0.0;

        if(algorithm==1){	
            output = signFunction(sum);
            if(instance.classAttribute().isNominal()){
                if(compare(output, 1.0) == 0){
                    output = 1.0;
                }else if (compare(output, -1.0) == 0){
                    output = 0.0;
                }
            }
        }else{
                output = sum;
                if(instance.classAttribute().isNominal()){
                        output = countOutput(sum, 0.0, 1.0);
                }
        }	
        return output;
    }
    
    public void evaluate(Instances data) throws Exception {
        if (this.classAttribute.isNominal()) {
            double correctlyClassifiedInstances = 0.0;
            Enumeration instances = data.enumerateInstances();
            while(instances.hasMoreElements()){
                    Instance instance = (Instance) instances.nextElement();
                    double retVal = classifyInstance(instance);
                    if (Double.compare(instance.classValue(),retVal)==0) {
                            correctlyClassifiedInstances+=1.0;
                    }
            }
            System.out.println("Accuracy = "+correctlyClassifiedInstances/(double)data.numInstances());
        }
        else { //numeric
            double deltaError = 0.0;
            Enumeration instances = data.enumerateInstances();
            while(instances.hasMoreElements()){
                    Instance instance = (Instance) instances.nextElement();
                    double retVal = classifyInstance(instance);
                    deltaError += Math.pow(retVal-instance.classValue(), 2);
            }
            System.out.println("MSE = "+ deltaError/2.0);
        }
    }
    
    public Attribute classAttribute() {
        return this.classAttribute;
    }
    
    public static Instances loadDatasetArff(String filePath) throws IOException { 
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(filePath));
        return loader.getDataSet();
    }
    
    public void setAlgo(int algorithm){
        this.algorithm = algorithm;
    }
    
    public static void main(String[] args) throws Exception {
        String dataset = "data/Zoo/zoo.data";

        Instances data = DataSource.read(dataset);
        data.setClass(data.attribute(data.numAttributes() - 1));

        SingleLayerPerceptron ptr = new SingleLayerPerceptron();
        ptr.setLearningRate(0.3);
        ptr.setThreshold(20);
        ptr.setMaxIteration(500);
        ptr.setInitialWeight(0.0);
        ptr.setIsNormalize(true);
        ptr.setAlgorithm(1);

//        ptr.buildClassifier(data);		
        WekaProcessor processor = new WekaProcessor();
        processor.readDataset(dataset);
        processor.buildClassifier(ptr);
        processor.nFoldCross_Eval(10);
        
//        System.out.println(ptr);

//        Evaluation eval = new Evaluation(data);
//        eval.evaluateModel(ptr, data);
//        System.out.println(eval.toSummaryString());
    }
    
}

