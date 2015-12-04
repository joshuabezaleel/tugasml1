/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author asus
 */
public class MyPTR extends Classifier{
    private Neuron neuron = new Neuron();
    private List<Double> inputList = new ArrayList<>();     // attrValue
    private int juminput;
    private double input;
    private int currEpoch = 1;
    private int maxEpoch;
    private double error;
    private double learningRate;
    private double momentum;
        
    public MyPTR() {
        maxEpoch = 10;
        learningRate = 0.1;
        momentum = 0.0;
    }
    
    public static int signFunction(double sum){
        int signResult = 0;
        if(sum>=0){
            signResult = 1;
        } else {
            signResult = -1;
        }
        return signResult;
    } 

    @Override
    public void buildClassifier(Instances data) throws Exception {
        NominalToBinary nomToBinFilter = new NominalToBinary();
        Normalize normalizeFilter = new Normalize();
        
        // PREPARE DATASET
        data = new Instances(data);
        data.deleteWithMissingClass();
        nomToBinFilter.setInputFormat(data);
        data = Filter.useFilter(data, nomToBinFilter);
        normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, normalizeFilter);
        
//        int jumtarget;              // jumlah instance
        List<Double> targetList = new ArrayList<>();
//        
        for(int j=0;j<(data.numAttributes())*data.numInstances();j++){
            inputList.add((double)0);
        }
//        for(int i=0;i<jumtarget;i++){
//            System.out.println("Target "+(i+1)+" : ");
//            input = reader.nextDouble();
//            targetList.add(input);
//        }
        
        juminput = data.numAttributes()-1;
        // juminput = jumlah atribut dlm satu instance
        neuron.initPreviousWeight(juminput+1,0);
        neuron.initCurrentWeight(juminput+1,0);
        neuron.initDeltaWeight(juminput+1,0);
        neuron.initPreviousDeltaWeight(juminput+1,0);
        //neuron.randomizeCurrentWeight();
        neuron.setCurrentWeightZero();
        
        double sum;
        
//        List<Double> errorSquareIterationAfterEpoch = new ArrayList<>();
        int errorEpoch = 0;
        
        int iteration;
        int h=0;
        do{
            //1 Epoch
            for(int i=0;i<data.numInstances();i++){
                sum = 0;
                System.out.println("epoch="+(h+1));
                System.out.println("iteration="+(i+1));
                //Hitung sum
                if(h==0){
                    //Inisialisasi bias
                    // debug
                    System.out.println("numInstance: "+data.numInstances());
                    System.out.println("numInput: "+inputList.size());
                    System.out.println("numAttr: "+data.numAttributes());
                    for(int j=0;j<data.numInstances();j++){
                        inputList.add((data.numAttributes()*j)+j, (double)1);
                    }
                    System.out.println("inputList: "+inputList.size());
                    //Input attribute value to inputList
                    for(int j=0;j<data.numInstances();j++){
                        for(int k=0;k<data.numAttributes()-1;k++){
                            inputList.add((k+1)+j*(data.numAttributes()+1),data.instance(j).value(k));
                        }
                    }
                }
                for(int j=i*juminput;j<(i+1)*juminput;j++){
                    System.out.println("neuron="+neuron.getCurrentWeight().get(j%4));
                    sum = sum + (inputList.get(j)*neuron.getCurrentWeight().get(j%4));
                }
                System.out.println("sum="+sum);
                error = data.instance(i).classValue() - signFunction(sum);
                System.out.println("targetlist="+data.instance(i).classValue());
                System.out.println("sign="+signFunction(sum));
                System.out.println("error="+error);
                for(int j=0;j<juminput;j++){
                    System.out.println("SetNewDeltaW"+j+"= "+inputList.get(j)+"*"+error+"*"+"learningRate");
                    neuron.setDeltaWeight(j, (inputList.get(j+(i*juminput))*error*learningRate)+(momentum*neuron.getPreviousDeltaWeight().get(j)));
                    neuron.setCurrentWeight(j, neuron.getCurrentWeight().get(j)+neuron.getDeltaWeight().get(j));
                    //System.out.println("new w"+j+"= "+neuron.getCurrentWeight().get(j));
                    System.out.println("new w"+j+"= "+neuron.getCurrentWeight().get(j));
                    neuron.setPreviousDeltaWeight(j, neuron.getDeltaWeight().get(j));
                }
            }
            double tempSum;
            double tempSumError = 0;
//            for(int i=0;i<data.numInstances();i++){
//                tempSum = 0;
//                for(int j=0;j<juminput;j++){
//                    tempSum = tempSum + inputList.get(j+(i*juminput));
//                }
////                errorSquareIterationAfterEpoch.set(i, Math.pow(targetList.get(i)-signFunction(tempSum),2));
//                tempSumError = tempSumError + Math.pow(targetList.get(i)-signFunction(tempSum),2);
//                errorEpoch = (int) (tempSumError * 0.5);
//            }
//            System.out.println("errorEpoch = "+errorEpoch);
            h++;
        }while(h<=maxEpoch && errorEpoch!=0);
    }
    
    @Override
    public double classifyInstance(Instance data) {
        List<Double> attributeValue = new ArrayList<>();
        //Inisialisasi bias
        attributeValue.add((double)1);
        
        //Input attribute value to inputList
        for(int k=0;k<data.numAttributes()-1;k++){
            inputList.add(data.value(k));
        }
        
        //including bias
        double tempSum = 0;
        for(int k=0;k<data.numAttributes();k++){
            tempSum = tempSum + (inputList.get(k)*neuron.getCurrentWeight().get(k));
        }
        
        return signFunction(tempSum);
    }
}
