/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

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
public class DeltaIncrementalCoba extends Classifier {
        Neuron neuron = new Neuron();
        List<Double> inputList = new ArrayList<>();
        List<Double> target = new ArrayList<>();
        List<Double> errorListEpoch = new ArrayList<>();
        int juminput;
        double input;
        int currEpoch = 1;
        int maxEpoch;
        double error;
        double learningRate;
        double momentum;
    
    public static int signFunction(double sum){
        int signResult = 0;
        if(sum>=0){
            signResult = 1;
        } else {
            signResult = -1;
        }
        return signResult;
    } 
       
    public DeltaIncrementalCoba() {
        maxEpoch = 10;
        learningRate = 0.1;
        momentum = 0.0;
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
                
                
        Scanner reader = new Scanner(System.in);
        
//        System.out.println("Jumlah input: ");
//        juminput = reader.nextInt();
//        
//        int jumtarget;
        List<Double> targetList = new ArrayList<>();
        
//        System.out.println("Jumlah target: ");
//        jumtarget = reader.nextInt();
        
        for(int j=0;j<(data.numAttributes()-1)*data.numInstances();j++){
            inputList.add((double)0);
        }
        
//        for(int i=0;i<jumtarget;i++){
//            System.out.println("Target "+(i+1)+" : ");
//            input = reader.nextDouble();
//            targetList.add(input);
//        }
        
         for(int i=0;i<data.numInstances();i++){
            errorListEpoch.add((double)0);
        }
         
        juminput = data.numAttributes()-1;
        neuron.initPreviousWeight(juminput,0);
        neuron.initCurrentWeight(juminput,0);
        neuron.initDeltaWeight(juminput,0);
        neuron.initPreviousDeltaWeight(juminput,0);
        //neuron.randomizeCurrentWeight();
        neuron.setCurrentWeightZero();
        
        double sum;
        double errorEpoch = 0;
        
        int iteration;
        int h=0;
        do{
            //1 Epoch
            for(int i=0;i<data.numInstances();i++){
                sum = 0;
                System.out.println("epoch="+(h+1));
                System.out.println("iteration="+(i+1));
                if(h==0){
                    //Inisialisasi bias
                    for(int j=0;j<data.numInstances();j++){
                        inputList.add(j*(data.numAttributes()-1), (double)1);
                    }
                    //Input attribute value to inputList
                    for(int j=0;j<data.numInstances();j++){
                        for(int k=0;k<data.numAttributes()-1;k++){
                            inputList.add((k+1)*j*(data.numAttributes()-1),data.instance(j+1).value(k+1));
                        }
                    }
                }
                //Hitung Sum
                for(int j=i*juminput;j<(i+1)*juminput;j++){
                    System.out.println("neuron="+neuron.getCurrentWeight().get(j%4));
                    sum = sum + (inputList.get(j)*neuron.getCurrentWeight().get(j%4));
                }
                System.out.println("sum="+sum);
                error = targetList.get(i) - sum;
                errorListEpoch.set(i, error);
                System.out.println("targetlist="+targetList.get(i));
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
            double tempSum = 0;
            double tempSumError = 0;
            for(int i=0;i<data.numInstances();i++){
//                tempSum = 0;
//                for(int j=0;j<juminput;j++){
//                    tempSum = tempSum + inputList.get(j+(i*juminput));
//                }
////                errorSquareIterationAfterEpoch.set(i, Math.pow(targetList.get(i)-signFunction(tempSum),2));
//                tempSumError = tempSumError + Math.pow(targetList.get(i)-tempSum,2);
//                errorEpoch = (int) (tempSumError * 0.5);
                tempSum = tempSum + Math.pow(errorListEpoch.get(i), 2);
            }
            errorEpoch = tempSum / 2;
            System.out.println("errorEpoch = "+errorEpoch);
            h++;
            System.out.println("==========End of Epoch==========");
        }while(h<=maxEpoch && errorEpoch!=0);
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        return 0.0;
    }
}
